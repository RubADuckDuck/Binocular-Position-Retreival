import mmap
import struct
import time
import math

# Define the size of the shared memory segment
shm_size = 1024

# Create a named shared memory file
with mmap.mmap(-1, shm_size, tagname='Local\\memfile', access=mmap.ACCESS_WRITE) as mm:
    t = 0
    while True:
        # Calculate oscillating coordinates
        x = math.sin(t)
        y = math.cos(t)
        z = math.sin(t) * math.cos(t)
        t += 0.1

        # Write data to shared memory
        mm.seek(0)
        # Pack the three coordinates as floats
        data = struct.pack('fff', x, y, z)
        mm.write(data)
        # Ensure data is written by flushing
        mm.flush()
        time.sleep(1/60)  # 60 times per second
