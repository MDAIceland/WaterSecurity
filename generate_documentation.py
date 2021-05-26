from nbconvert.exporters.exporter_locator import E
import pdoc
import os
from nbconvert import HTMLExporter
import nbformat


def _fix_module_names(module_name):
    return ".".join((["WaterSecurity"] + module_name.split(".")[1:]))


def main():
    """
    Generates Project Documentation Based on DocStrings
    """
    context = pdoc.Context()

    with open(".docignore", "r") as rf:
        ignore_strings = rf.read().splitlines()

    modules = pdoc.Module(
        ".",
        context=context,
        skip_errors=True,
        docfilter=lambda x: _fix_module_names(x.name) not in ignore_strings,
    )

    pdoc.link_inheritance(context)

    def recursive_htmls(mod):
        yield mod.name, mod.html(), bool(mod.submodules())
        for submod in mod.submodules():
            yield from recursive_htmls(submod)

    for module_name, html, has_subm in recursive_htmls(modules):
        module_name = _fix_module_names(module_name)
        if has_subm:
            fname = f"docs/{os.sep.join(module_name.split('.'))}/index.html"
        else:
            fname = f"docs/{os.sep.join(module_name.split('.'))}.html"
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w", encoding="utf-8") as f:
            f.writelines(html)

    html_exporter = HTMLExporter()
    html_exporter.template_name = "classic"
    nb_dir = os.path.join("docs", "WaterSecurity", "notebooks")
    os.makedirs(nb_dir, exist_ok=True)
    for subdir, dirs, files in os.walk(r"."):
        if any(
            (x != ".") and ((len(x) > 1) and ((x[0] in ("_", ".")) or ("venv" in x)))
            for x in subdir.split(os.sep)
        ):
            continue
        for filename in files:
            filepath = os.path.join(subdir, filename)
            if filepath.endswith(".ipynb"):
                with open(filepath, "r", encoding="utf-8") as fp:
                    (body, resources) = html_exporter.from_notebook_node(
                        nbformat.read(fp, as_version=4)
                    )
                with open(
                    os.path.join(
                        nb_dir, os.path.basename(filepath).split(".")[0] + ".html"
                    ),
                    "w",
                    encoding="utf-8",
                ) as exp:
                    exp.writelines(body)


if __name__ == "__main__":
    main()
