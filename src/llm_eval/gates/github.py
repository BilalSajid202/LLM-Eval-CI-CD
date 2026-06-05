"""GitHub commit status integration for merge gates."""

from __future__ import annotations

from uuid import UUID

import httpx

from llm_eval.gates.evaluator import GateEvaluator
from llm_eval.models.types import EvalRun, GateResult


class GitHubStatusPoster:
    CONTEXT = "llm-eval/quality-gates"

    def __init__(self, token: str, repository: str, dashboard_url: str = ""):
        self.token = token
        self.repository = repository
        self.dashboard_url = dashboard_url.rstrip("/")

    def post_status(
        self,
        sha: str,
        state: str,
        description: str,
        run_id: UUID | None = None,
    ) -> bool:
        if not self.token or not self.repository:
            return False

        target_url = f"{self.dashboard_url}/runs/{run_id}" if run_id and self.dashboard_url else ""

        response = httpx.post(
            f"https://api.github.com/repos/{self.repository}/statuses/{sha}",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
            },
            json={
                "state": state,
                "description": description[:140],
                "context": self.CONTEXT,
                "target_url": target_url or None,
            },
            timeout=30,
        )
        return response.status_code == 201

    def post_from_run(self, run: EvalRun, sha: str | None = None) -> bool:
        from llm_eval.config.loader import load_eval_config

        sha = sha or run.git_sha
        if run.gate_results:
            ev = GateEvaluator(load_eval_config())
            state, description = ev.overall_state(run.gate_results)
        else:
            state, description = "pending", "Eval in progress"
        return self.post_status(sha, state, description, run.run_id)

    def post_pr_comment(self, pr_number: int, body: str) -> bool:
        if not self.token or not self.repository:
            return False
        response = httpx.post(
            f"https://api.github.com/repos/{self.repository}/issues/{pr_number}/comments",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
            },
            json={"body": body},
            timeout=30,
        )
        return response.status_code == 201
