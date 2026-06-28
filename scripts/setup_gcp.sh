#!/usr/bin/env bash
# ── HealthBridgeAI GCP Infrastructure Setup ──────────────────────────────────
# Run this ONCE before the first Cloud Build / Cloud Run deployment.
# Safe to re-run — most operations are idempotent.
#
# Prerequisites:
#   gcloud CLI installed and authenticated (gcloud auth login)
#   Owner or Editor + required roles on the target project
#
# Usage:
#   export PROJECT_ID=your-gcp-project-id
#   export REGION=us-central1          # optional, default us-central1
#   bash scripts/setup_gcp.sh
#
# After this script:
#   1. Fill in every secret value:  bash scripts/setup_gcp.sh --set-secrets
#   2. Create Pinecone index:       python scripts/setup_pinecone.py
#   3. Trigger Cloud Build:         gcloud builds submit --config deploy/cloudbuild.yaml
#   4. Update Pub/Sub push URL:     bash scripts/setup_gcp.sh --update-push-url
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID env var}"
REGION="${REGION:-us-central1}"
AR_REPO="healthbridge"                  # Artifact Registry repo name
BUCKET="healthbridge-assets"            # GCS corpus + audio bucket
FIRESTORE_DB="(default)"
TOPIC="healthbridge-inbound"
SUBSCRIPTION="healthbridge-inbound-sub"
SA_WEBHOOK="healthbridge-webhook"
SA_PROCESSOR="healthbridge-processor"
SA_CLOUDBUILD="healthbridge-cloudbuild"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# ── Mode flags ────────────────────────────────────────────────────────────────
MODE="${1:-}"

# ─────────────────────────────────────────────────────────────────────────────
set_secrets_mode() {
  info "Prompting for secret values..."
  set_secret() {
    local name="$1" prompt="$2"
    echo -n "  $prompt: "
    read -rs value; echo
    if [[ -z "$value" ]]; then
      warn "Skipping $name (empty)"
      return
    fi
    echo -n "$value" | gcloud secrets versions add "$name" \
      --project="$PROJECT_ID" --data-file=-
    info "Updated secret: $name"
  }

  set_secret "whatchamp-api-key"            "WHATCHAMP_API_KEY"
  set_secret "whatchamp-phone-number-id"    "WHATCHAMP_PHONE_NUMBER_ID (Meta phone_number_id)"
  set_secret "whatchamp-phone-number"       "WHATCHAMP_PHONE_NUMBER (E.164, e.g. +2348012345678)"
  set_secret "whatchamp-webhook-secret"     "WHATCHAMP_WEBHOOK_SECRET (Meta app secret)"
  set_secret "whatchamp-webhook-verify-token" "WHATCHAMP_WEBHOOK_VERIFY_TOKEN (your chosen verify token)"
  set_secret "pinecone-api-key"             "PINECONE_API_KEY"
  set_secret "openrouter-api-key"           "OPENROUTER_API_KEY"
  set_secret "tavily-api-key"               "TAVILY_API_KEY"
  set_secret "huggingface-token"            "HUGGINGFACE_TOKEN (optional, press Enter to skip)"
  set_secret "yarngpt-api-key"             "YARNGPT_API_KEY (optional, press Enter to skip)"
  set_secret "natlas-api-key"             "NATLAS_API_KEY (optional, press Enter to skip)"
  info "Secrets updated."
  exit 0
}

update_push_url_mode() {
  info "Fetching processor Cloud Run URL..."
  PROCESSOR_URL=$(gcloud run services describe healthbridge-processor \
    --region="$REGION" --project="$PROJECT_ID" \
    --format="value(status.url)")
  if [[ -z "$PROCESSOR_URL" ]]; then
    warn "Processor service not found — deploy it first via Cloud Build"
    exit 1
  fi
  PUSH_URL="${PROCESSOR_URL}/process"
  info "Updating Pub/Sub push subscription → $PUSH_URL"
  gcloud pubsub subscriptions modify-push-config "$SUBSCRIPTION" \
    --project="$PROJECT_ID" \
    --push-endpoint="$PUSH_URL" \
    --push-auth-service-account="${SA_PROCESSOR}@${PROJECT_ID}.iam.gserviceaccount.com"
  info "Push URL updated."
  exit 0
}

[[ "$MODE" == "--set-secrets"    ]] && set_secrets_mode
[[ "$MODE" == "--update-push-url" ]] && update_push_url_mode

# ─────────────────────────────────────────────────────────────────────────────
# Full provisioning
# ─────────────────────────────────────────────────────────────────────────────
info "=== HealthBridgeAI GCP Setup  project=$PROJECT_ID  region=$REGION ==="

gcloud config set project "$PROJECT_ID" --quiet

# ── 1. Enable APIs ────────────────────────────────────────────────────────────
info "Enabling GCP APIs..."
gcloud services enable \
  run.googleapis.com \
  pubsub.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  firestore.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project="$PROJECT_ID" --quiet

# ── 2. Artifact Registry ──────────────────────────────────────────────────────
info "Creating Artifact Registry repo: $AR_REPO..."
gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --project="$PROJECT_ID" \
  --description="HealthBridgeAI container images" \
  --quiet 2>/dev/null || info "  (already exists)"

AR_HOST="${REGION}-docker.pkg.dev"
AR_PATH="${AR_HOST}/${PROJECT_ID}/${AR_REPO}"
info "  Registry: $AR_PATH"

# ── 3. GCS bucket ─────────────────────────────────────────────────────────────
info "Creating GCS bucket: gs://$BUCKET..."
gcloud storage buckets create "gs://$BUCKET" \
  --project="$PROJECT_ID" \
  --location="$REGION" \
  --uniform-bucket-level-access \
  --quiet 2>/dev/null || info "  (already exists)"

# Create directory structure markers
for prefix in "corpora/" "audio/" "knowledge-bases/tb/" "knowledge-bases/hiv/" "knowledge-bases/malaria/"; do
  echo "" | gcloud storage cp - "gs://${BUCKET}/${prefix}.keep" --quiet 2>/dev/null || true
done

# ── 4. Firestore ──────────────────────────────────────────────────────────────
info "Creating Firestore database: $FIRESTORE_DB..."
gcloud firestore databases create \
  --project="$PROJECT_ID" \
  --location="$REGION" \
  --type=firestore-native \
  --quiet 2>/dev/null || info "  (already exists)"

# ── 5. Service accounts ───────────────────────────────────────────────────────
create_sa() {
  local name="$1" display="$2"
  gcloud iam service-accounts create "$name" \
    --display-name="$display" \
    --project="$PROJECT_ID" \
    --quiet 2>/dev/null || info "  SA $name already exists"
}

info "Creating service accounts..."
create_sa "$SA_WEBHOOK"    "HealthBridgeAI Webhook Service"
create_sa "$SA_PROCESSOR"  "HealthBridgeAI Processor Service"
create_sa "$SA_CLOUDBUILD" "HealthBridgeAI Cloud Build"

WEBHOOK_SA="${SA_WEBHOOK}@${PROJECT_ID}.iam.gserviceaccount.com"
PROCESSOR_SA="${SA_PROCESSOR}@${PROJECT_ID}.iam.gserviceaccount.com"
CLOUDBUILD_SA="${SA_CLOUDBUILD}@${PROJECT_ID}.iam.gserviceaccount.com"

# ── 6. IAM bindings ───────────────────────────────────────────────────────────
info "Granting IAM roles..."
bind() {
  local sa="$1" role="$2"
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$sa" --role="$role" --quiet >/dev/null
}

# Webhook SA — publishes to Pub/Sub, reads secrets
bind "$WEBHOOK_SA"  "roles/pubsub.publisher"
bind "$WEBHOOK_SA"  "roles/secretmanager.secretAccessor"

# Processor SA — reads Pub/Sub, reads/writes Firestore + GCS, reads secrets
bind "$PROCESSOR_SA" "roles/pubsub.subscriber"
bind "$PROCESSOR_SA" "roles/datastore.user"
bind "$PROCESSOR_SA" "roles/storage.objectAdmin"
bind "$PROCESSOR_SA" "roles/secretmanager.secretAccessor"
# Processor SA needs to receive push from Pub/Sub (iam.serviceAccounts.actAs)
bind "$PROCESSOR_SA" "roles/iam.serviceAccountTokenCreator"

# Cloud Build SA — builds + deploys Cloud Run + pushes to Artifact Registry
bind "$CLOUDBUILD_SA" "roles/run.admin"
bind "$CLOUDBUILD_SA" "roles/artifactregistry.writer"
bind "$CLOUDBUILD_SA" "roles/iam.serviceAccountUser"
bind "$CLOUDBUILD_SA" "roles/secretmanager.secretAccessor"
bind "$CLOUDBUILD_SA" "roles/storage.objectViewer"

# Pub/Sub service account needs to create auth tokens for push
PUBSUB_SA="service-$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')@gcp-sa-pubsub.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$PUBSUB_SA" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --quiet >/dev/null

# ── 7. Pub/Sub topic + push subscription ─────────────────────────────────────
info "Creating Pub/Sub topic: $TOPIC..."
gcloud pubsub topics create "$TOPIC" \
  --project="$PROJECT_ID" --quiet 2>/dev/null || info "  (already exists)"

# Dead-letter topic
gcloud pubsub topics create "${TOPIC}-dead-letter" \
  --project="$PROJECT_ID" --quiet 2>/dev/null || info "  (dead-letter already exists)"

info "Creating Pub/Sub push subscription: $SUBSCRIPTION..."
# Push URL is a placeholder — update after first Cloud Run deployment:
#   bash scripts/setup_gcp.sh --update-push-url
PLACEHOLDER_URL="https://placeholder.run.app/process"
gcloud pubsub subscriptions create "$SUBSCRIPTION" \
  --topic="$TOPIC" \
  --project="$PROJECT_ID" \
  --push-endpoint="$PLACEHOLDER_URL" \
  --push-auth-service-account="$PROCESSOR_SA" \
  --ack-deadline=300 \
  --min-retry-delay=10s \
  --max-retry-delay=600s \
  --dead-letter-topic="${TOPIC}-dead-letter" \
  --max-delivery-attempts=5 \
  --quiet 2>/dev/null || info "  (already exists — run --update-push-url after deployment)"

# Dead-letter subscription (pull, for inspection)
gcloud pubsub subscriptions create "${SUBSCRIPTION}-dead-letter-pull" \
  --topic="${TOPIC}-dead-letter" \
  --project="$PROJECT_ID" \
  --quiet 2>/dev/null || true

# ── 8. Secret Manager secrets (empty shells) ──────────────────────────────────
info "Creating Secret Manager secrets (empty — fill with --set-secrets)..."
create_secret() {
  local name="$1"
  gcloud secrets create "$name" \
    --project="$PROJECT_ID" \
    --replication-policy=automatic \
    --quiet 2>/dev/null || true   # ignore if already exists
}

create_secret "whatchamp-api-key"
create_secret "whatchamp-phone-number-id"
create_secret "whatchamp-phone-number"
create_secret "whatchamp-webhook-secret"
create_secret "whatchamp-webhook-verify-token"
create_secret "pinecone-api-key"
create_secret "openrouter-api-key"
create_secret "tavily-api-key"
create_secret "huggingface-token"
create_secret "yarngpt-api-key"
create_secret "natlas-api-key"

# ── 9. Cloud Build trigger (optional — requires repo connected) ───────────────
info "Skipping Cloud Build trigger — connect the repo in Cloud Build UI first,"
info "  then run: gcloud builds triggers create github ..."

# ── 10. Summary ───────────────────────────────────────────────────────────────
echo ""
info "=== Setup complete ==="
echo ""
echo "  Artifact Registry : $AR_PATH"
echo "  GCS bucket        : gs://$BUCKET"
echo "  Pub/Sub topic     : $TOPIC"
echo "  Subscription      : $SUBSCRIPTION  (push URL = placeholder)"
echo ""
echo "  Next steps:"
echo "  1. Fill secrets:      bash scripts/setup_gcp.sh --set-secrets"
echo "  2. Create Pinecone:   python scripts/setup_pinecone.py"
echo "  3. Run Cloud Build:   gcloud builds submit --config deploy/cloudbuild.yaml \\"
echo "       --substitutions _GCP_PROJECT=${PROJECT_ID},_REGION=${REGION},_REPO=${AR_PATH}"
echo "  4. Update push URL:   bash scripts/setup_gcp.sh --update-push-url"
echo "  5. Index knowledge:   python scripts/populate_kb.py"
echo "  6. Warm cache:        python scripts/warm_cache.py"
