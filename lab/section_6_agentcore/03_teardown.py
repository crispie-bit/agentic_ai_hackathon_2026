"""
TAKE-HOME 4, part 2 — delete everything lab 07 created.

    uv run section_6_agentcore/03_teardown.py             # DRY RUN: shows what would go
    uv run section_6_agentcore/03_teardown.py --yes       # actually delete it
    uv run section_6_agentcore/03_teardown.py --list      # what is deployed in this region?

This is the more important half of the lab. An agent you can deploy but not
reliably delete is a recurring bill and someone else's cleanup ticket.

Deliberately safe by default: with no flags it only *reports*. You must pass
--yes to destroy anything.

WHAT GETS DELETED — measured on a real direct_code_deploy teardown, which
reported exactly six things:

  - the AgentCore runtime (the DEFAULT endpoint goes with it automatically)
  - the two S3 artifacts under the agent's prefix: source.zip, deployment.zip
  - the IAM execution role — but only the auto-created one, and only if no
    other agent references it
  - the agent's entry in .bedrock_agentcore.yaml
  - the yaml file itself, once no agents remain in it

For a container deploy it would also remove the CodeBuild project and the images
in the agent's ECR repository. A direct_code_deploy agent has neither, so the
teardown explicitly logs "Skipping CodeBuild cleanup for direct_code_deploy".

WHAT SURVIVES, AND WHY
  - the shared S3 source bucket (bedrock-agentcore-codebuild-sources-<acct>-<region>).
    Per-account, not per-agent — other agents use it. Only your prefix is emptied.
  - CloudWatch log groups, so you can still read the traces afterwards
  - the ECR *repository*, unless you pass --delete-ecr-repo (container only)
  - an execution role another agent still references

# ======================================================================
# TODO(student) 1 — Run the dry run first and read the list. Then run --yes
#   and diff the two outputs. Anything reported but not deleted is something
#   you now have to clean up by hand — know which those are.
#
# TODO(student) 2 — After teardown, run 07 with --status. Work out from the
#   error what state the yaml is now in.
#
# TODO(student) 3 — Check the AWS console: Bedrock AgentCore runtimes, IAM
#   roles matching AmazonBedrockAgentCoreSDKRuntime-*, and S3 for
#   bedrock-agentcore-codebuild-sources-*. Confirm what is really gone.
# ======================================================================

"""

import argparse
import os
import sys

import _bootstrap  # noqa: F401

from _agentcore import (CONFIG, DEFAULT_NAME, REGION, attach_runtime, banner,
                        configured_agents, list_remote, wait_gone)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Delete the AgentCore resources created by lab 07.")
    ap.add_argument("--agent", default=os.environ.get("AGENTCORE_AGENT_NAME",
                                                      DEFAULT_NAME),
                    help=f"runtime name (default: {DEFAULT_NAME})")
    ap.add_argument("--yes", action="store_true",
                    help="actually delete. Without this it is a dry run.")
    ap.add_argument("--delete-ecr-repo", action="store_true",
                    help="also delete the ECR repository (container deploys)")
    ap.add_argument("--list", action="store_true",
                    help="list runtimes in the region and exit")
    args = ap.parse_args()

    if args.list:
        list_remote()
        return 0

    known = configured_agents()
    if known and args.agent not in known:
        print(f"'{args.agent}' is not in {CONFIG.name}.")
        print(f"  configured here: {', '.join(known)}")
        print(f"\n  Pass one of those with --agent, or use --list to see what is "
              f"actually deployed in {REGION}.")
        return 1
    if not known:
        print(f"No {CONFIG.name} found — nothing was deployed from this checkout.")
        print("  Use --list to check the account for orphaned runtimes.")
        return 1

    # attach_runtime() binds a fresh Runtime to the existing config — see the
    # module docstring in _agentcore.py for why that needs a shim.
    rt = attach_runtime(args.agent)

    mode = "DELETING" if args.yes else "DRY RUN — nothing will be deleted"
    banner(f"{mode}   agent={args.agent}   region={REGION}")

    try:
        result = rt.destroy(dry_run=not args.yes,
                            delete_ecr_repo=args.delete_ecr_repo)
    except Exception as exc:
        print(f"  destroy failed: {type(exc).__name__}: {exc}")
        print("\n  If this says the agent is not found, it is already gone —")
        print("  confirm with: uv run section_6_agentcore/03_teardown.py --list")
        return 1

    raw = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    removed = raw.get("resources_removed") or raw.get("removed") or []
    warnings = raw.get("warnings") or []
    errors = raw.get("errors") or []

    if removed:
        print(f"  {'would remove' if not args.yes else 'removed'}:")
        for r in removed:
            print(f"    - {r}")
    else:
        print("  nothing to remove (already torn down?)")
    for w in warnings:
        print(f"  ! {w}")
    for e in errors:
        print(f"  ERROR {e}")

    if not args.yes:
        print(f"\n  This was a DRY RUN. To actually delete:")
        print(f"    uv run section_6_agentcore/03_teardown.py --agent {args.agent} --yes")
        return 0

    banner("VERIFY")
    wait_gone(args.agent)
    list_remote()

    print("""
  Still in your account on purpose:
    - the shared S3 source bucket (per-account, other agents may use it)
    - CloudWatch log groups, so the traces remain readable
    - the ECR repository, unless you passed --delete-ecr-repo
      (a direct_code_deploy agent never creates one)

  Deleting the runtime stops the charges that matter. The leftovers above cost
  cents, but knowing WHICH resources a teardown does not touch is the actual
  lesson here — that list is what turns into a surprise bill on a real project.""")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
