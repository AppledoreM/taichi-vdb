import taichi as ti


def align_size(x, align):
    return (x + (align - 1)) & ~(align - 1)


def vdb_assert(expr: bool, log_str: str):
    assert expr, ">>>> [VDB]: " + log_str


def vdb_log(log_str: str):
    print(">> [VDB]: {}".format(log_str))


#@param x, y      - denotes the parameter to compare
#       tolerence - denotes the tolerence of comparison
#@detail: Returns if |x - y| < tolerence
@ti.func
def approx_equal(x, y, tolerence) -> bool:
    return ti.abs(x - y) < tolerence


if __name__ == "__main__":
    pass
