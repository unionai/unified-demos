from dataclasses import dataclass

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
def test2() -> testOut:
    return True

@union.workflow
def wf():
    r=test()
    r2=test2()
    print(r)

if __name__ == "__main__":
    wf()