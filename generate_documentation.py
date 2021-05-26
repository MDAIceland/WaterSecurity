from nbconvert.exporters.exporter_locator import E
import pdoc
import os
from nbconvert import HTMLExporter
import nbformat

context = pdoc.Context()

with open(".docignore","r") as rf:
    ignore_strings = rf.read().splitlines()

modules = pdoc.Module(".", context=context, skip_errors=True, docfilter=lambda x: x.name not in ignore_strings)
        
pdoc.link_inheritance(context)

def recursive_htmls(mod):
    yield mod.name, mod.html(), bool(mod.submodules())
    for submod in mod.submodules():
        yield from recursive_htmls(submod)

for module_name, html, has_subm in recursive_htmls(modules):
    print(module_name)
    if has_subm:
        fname = f"docs/{'/'.join(module_name.split('.'))}/index.html"
    else:
        fname = f"docs/{'/'.join(module_name.split('.'))}.html"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname,"w", encoding="utf-8") as f:
        f.writelines(html)

html_exporter = HTMLExporter()
html_exporter.template_name = 'classic'

for subdir, dirs, files in os.walk(r'.'):
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith(".ipynb"):
            with open(filepath, "r",encoding="utf-8") as fp:
                (body, resources) = html_exporter.from_notebook_node(nbformat.read(fp, as_version=4))
            with open("docs/WaterSecurity/notebooks/"+os.path.basename(filepath).split(".")[0]+".html","w",encoding="utf-8") as exp:   
                exp.writelines(body)
