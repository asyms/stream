from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance

my_memory_instance = MemoryInstance(
    name="hi", size=123456 * 8, r_bw=64, w_bw=64, auto_cost_extraction=True
)

print(my_memory_instance.r_cost)
