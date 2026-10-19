# Using senso-check as a Claude Code SessionStart hook (example)

This prints contradicted manifest claims at the start of every agent session. Add it to your project's
`.claude/settings.json` yourself; nothing in this repository edits a settings file.

First install the tool once (clone, venv, `pip install -r requirements.txt`, `.env` with the NIM keys; see the README).
The hook must run from a directory whose `.env` holds the keys, or the keys must be set in the environment.
Replace `/path/to/senso-check` and `owner/name`.

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cd /path/to/senso-check && .venv/bin/python senso_check.py --repo owner/name --local-path \"$CLAUDE_PROJECT_DIR\" --skip-explain 2>/dev/null | grep -A2 contradicted || true",
            "timeout": 180
          }
        ]
      }
    ]
  }
}
```

The hook's stdout is added to the session context. Text mode prints each contradicted claim with the rule and the
measured fact on the next two lines (`grep -A2`), so the agent sees which manifest lines to distrust. A first run takes
1 to 2 minutes (the extraction calls dominate), so use a larger `timeout` for long manifests; later runs reuse the cached
extraction in `senso-out/` and take a few seconds. Not tested inside Claude Code; the command itself was tested from a shell.
Remember that a contradicted result means "check this line", not "proven error" (see Limitations in the README).
