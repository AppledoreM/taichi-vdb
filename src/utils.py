import taichi as ti


def align_size(x, align):
    return (x + (align - 1)) & ~(align - 1)


def vdb_assert(expr: bool, log_str: str):
    assert expr, ">>>> [VDB]: " + log_str


def vdb_log(log_str: str):
    print(">> [VDB]: {}".format(log_str))




if __name__ == "__main__":
    pass
