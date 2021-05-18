import pdoc
import os
context = pdoc.Context()

modules = pdoc.Module(".", context=context, skip_errors=True)
        
pdoc.link_inheritance(context)

def recursive_htmls(mod):
    yield mod.name, mod.html()
    for submod in mod.submodules():
        yield from recursive_htmls(submod)

for module_name, html in recursive_htmls(modules):
    fname = f"documentation/{'/'.join(module_name.split('.'))}/index.html"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname,"w", encoding="utf-8") as f:
        f.writelines(html)

