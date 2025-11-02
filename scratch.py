from dataclasses import dataclass
from flytekit import conditional
import union

@dataclass
class testOut:
    var1: bool
    var2: str

image = union.ImageSpec(
    builder="union",
    base_image="ghcr.io/unionai-oss/union:py3.10-latest",
    name="unified_demo1",
    packages=["scikit-learn", "datasets", "pandas",
              "union", "flytekitplugins-spark", "delta-sharing",
              "tabulate", "flytekitplugins-deck-standard"],

)

@union.task(
    container_image=image
)
def test() -> testOut:
    return testOut(True,"test")

@union.task(
    container_image=image
)
def test2() -> bool:
    return True

@union.workflow
def wf():
    r=test()
    r2=test2()
    print(r)
    r_bool = r.var1
    print(r.var2)
    b = r.var1 == True
    conditional("test").if_(1==1).then(print("true"))

if __name__ == "__main__":
    wf()