"""Keep operational errors, suppress per-frame geometry diagnostics by default."""
import builtins
import os

_PREFIXES = ('[FITQ]', '[SHAFT_SELECT]', '[SCORE_GEOM]', '[SCORE_DEBUG]',
             '[fuse]', '[PERF]', '[dart] [', '[dart] fused:', '[dart] hit debug:',
             '[dart] HIT ', '[dart] AUTO HIT ', '[dart] idle', '[dart] armed',
             '[dart] empty', '[dart] probe')

def print(*args, **kwargs):
    text=str(args[0]) if args else ''
    if os.environ.get('SMARTDARTS_VERBOSE','0')!='1' and text.startswith(_PREFIXES):
        return
    builtins.print(*args, **kwargs)
