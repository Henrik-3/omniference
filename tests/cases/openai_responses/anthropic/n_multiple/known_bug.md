# `n: 2` is silently ignored for Anthropic (Responses)

Responses has no `n` parameter and Anthropic returns one completion; the unknown field is ignored and the request succeeds. Low severity: reported to document the matrix cell.
Snapshots record current behaviour.
