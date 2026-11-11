"""Prepare/collect actual exact-code Julia execution; not an IJulia-kernel claim."""
import base64
import hashlib
import json
import sys
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/"notebooks/Intro_English_GPT.ipynb"
SCRATCH=ROOT/"tmp/english_notebook_validation"
notebook=nbformat.read(NB,as_version=4)
indices=[i for i,c in enumerate(notebook.cells) if c.cell_type=="code"]

if "--collect" not in sys.argv:
    SCRATCH.mkdir(parents=True,exist_ok=True)
    for i in indices:
        for ext in ("done","txt","png"):
            (SCRATCH/f"cell_{i:02d}.{ext}").unlink(missing_ok=True)
        (SCRATCH/f"cell_{i:02d}.jl").write_text(notebook.cells[i].source+"\n",encoding="utf-8")
    runner='''cd(normpath(joinpath(@__DIR__,"..","..","notebooks")))
for idx in INDICES
    stem=joinpath(@__DIR__,"cell_"*lpad(string(idx),2,'0'))
    println("Executing English notebook cell ",idx)
    flush(stdout)
    try
        open(stem*".txt","w") do io
            redirect_stdout(io) do
                redirect_stderr(io) do
                    result=Base.include(Main,stem*".jl")
                    if isdefined(Main,:Plots) && result isa Main.Plots.Plot
                        Base.invokelatest(Main.Plots.savefig,result,stem*".png")
                    end
                end
            end
        end
        write(stem*".done","ok")
    catch err
        showerror(stderr,err,catch_backtrace())
        rethrow()
    end
end
println("All exact Julia English notebook cells passed.")
'''.replace("INDICES",repr(indices))
    (SCRATCH/"run.jl").write_text(runner,encoding="utf-8")
    print(f"Prepared {len(indices)} Julia code cells")
else:
    figures=0
    for count,i in enumerate(indices,1):
        stem=SCRATCH/f"cell_{i:02d}"
        assert stem.with_suffix(".done").read_text()=="ok"
        assert stem.with_suffix(".jl").read_text(encoding="utf-8")==notebook.cells[i].source+"\n"
        outputs=[]
        text=stem.with_suffix(".txt").read_text(encoding="utf-8")
        if text:
            outputs.append(nbformat.v4.new_output("stream",name="stdout",text=text))
        if stem.with_suffix(".png").exists():
            encoded=base64.b64encode(stem.with_suffix(".png").read_bytes()).decode()
            outputs.append(nbformat.v4.new_output("display_data",data={"image/png":encoded},metadata={}))
            figures+=1
        notebook.cells[i].outputs=outputs
        notebook.cells[i].execution_count=count
    method="Exact saved Julia code cells executed sequentially in a fresh Julia process; not an IJulia-kernel run"
    notebook.metadata["validation"]={"method":method,"code_cells":len(indices),"embedded_figures":figures}
    nbformat.validate(notebook)
    nbformat.write(notebook,NB)
    report=dict(notebook.metadata.validation)
    report["notebook_sha256"]=hashlib.sha256(NB.read_bytes()).hexdigest()
    report["code_cell_sha256"]=[hashlib.sha256(notebook.cells[i].source.encode()).hexdigest() for i in indices]
    (ROOT/"output/english_basics/notebook_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    from nbconvert import HTMLExporter
    html,_=HTMLExporter(template_name="lab").from_notebook_node(notebook)
    (SCRATCH/"preview.html").write_text(html,encoding="utf-8")
    print(f"Embedded {len(indices)} executed cells and {figures} figures; {NB.stat().st_size:,} bytes")
