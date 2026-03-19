import traceback
import sys
import torch
import transformers.utils.import_utils
from packaging import version
v = version.parse(torch.__version__.split('+')[0])
transformers.utils.import_utils.is_torch_greater_than_1_12 = lambda: v >= version.parse("1.12")
transformers.utils.import_utils.is_torch_greater_than_1_13 = lambda: v >= version.parse("1.13")
import transformers.utils
transformers.utils.is_torch_greater_than_1_12 = transformers.utils.import_utils.is_torch_greater_than_1_12
transformers.utils.is_torch_greater_than_1_13 = transformers.utils.import_utils.is_torch_greater_than_1_13

try:
    from cloner_engine import ClonerEngine
    import asyncio
    
    async def diag():
        try:
            ce = ClonerEngine()
            await ce.warm_up()
            print("Warm up success")
        except Exception:
            traceback.print_exc()
            
    asyncio.run(diag())
except Exception:
    traceback.print_exc()
