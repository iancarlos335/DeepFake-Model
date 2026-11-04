"""Executes a notebook's code cells in order in a single Python process (no Jupyter kernel /
sockets needed) and stores stdout, displayed tables and matplotlib figures as cell outputs."""
import ast, base64, io, sys, time, traceback, contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_output

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
ns = {"__name__": "__main__"}
outs = []


def _rich(obj):
    data = {"text/plain": repr(obj)}
    if hasattr(obj, "_repr_html_"):
        try:
            h = obj._repr_html_()
            if h: data["text/html"] = h
        except Exception:
            pass
    return data


def display(*objs):
    for o in objs:
        outs.append(new_output("display_data", data=_rich(o)))


ns["display"] = display
n_exec, t_all = 0, time.time()
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    n_exec += 1
    outs = []
    buf = io.StringIO()
    t0 = time.time()
    try:
        tree = ast.parse(cell.source)
        last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        body = ast.Module(body=tree.body[:-1] if last else tree.body, type_ignores=[])
        with contextlib.redirect_stdout(buf):
            exec(compile(body, f"<cell {n_exec}>", "exec"), ns)
            val = eval(compile(ast.Expression(last.value), f"<cell {n_exec}>", "eval"), ns) if last else None
        text = buf.getvalue()
        cell_outs = [new_output("stream", name="stdout", text=text)] if text else []
        cell_outs += outs
        for num in plt.get_fignums():
            f = plt.figure(num); b = io.BytesIO(); f.savefig(b, format="png", dpi=110, bbox_inches="tight")
            cell_outs.append(new_output("display_data", data={"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}))
        plt.close("all")
        if val is not None:
            cell_outs.append(new_output("execute_result", data=_rich(val), execution_count=n_exec))
        cell.outputs = cell_outs
        cell.execution_count = n_exec
        print(f"cell {n_exec} ok {time.time() - t0:.1f}s", flush=True)
    except Exception as e:
        cell.outputs = [new_output("stream", name="stdout", text=buf.getvalue()),
                        new_output("error", ename=type(e).__name__, evalue=str(e), traceback=traceback.format_exc().splitlines())]
        cell.execution_count = n_exec
        nbformat.write(nb, path)
        print(f"cell {n_exec} FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
nbformat.write(nb, path)
print(f"all {n_exec} code cells ok in {time.time() - t_all:.0f}s")
