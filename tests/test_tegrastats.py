from benchmark.parse_tegrastats import parse, summarise

NANO = [
    "RAM 2489/3956MB (lfb 120x4MB) SWAP 0/1978MB (cached 0MB) CPU [35%@1479,28%@1479,30%@1479,33%@1479] "
    "EMC_FREQ 18%@1600 GR3D_FREQ 76%@921 APE 25 PLL@32C CPU@38.5C PMIC@100C GPU@36C AO@44.5C "
    "thermal@37.25C POM_5V_IN 6123/5900 POM_5V_GPU 2400/2300 POM_5V_CPU 1100/1050",
    "RAM 2501/3956MB (lfb 118x4MB) SWAP 0/1978MB (cached 0MB) CPU [40%@1479,25%@1479,31%@1479,36%@1479] "
    "EMC_FREQ 19%@1600 GR3D_FREQ 84%@921 APE 25 PLL@32C CPU@39C PMIC@100C GPU@37C AO@45C "
    "thermal@37.5C POM_5V_IN 6477/5950 POM_5V_GPU 2600/2350 POM_5V_CPU 1150/1060",
]
ORIN = ["RAM 3200/7620MB SWAP 0/3810MB CPU [10%@1510,5%@1510] GR3D_FREQ 60% cpu@45C gpu@44.5C "
        "VDD_IN 7012mW/6900mW VDD_CPU_GPU_CV 1800mW/1700mW"]


def test_nano_format():
    s = summarise(parse(NANO), fps=25.0)
    assert s["samples"] == 2
    assert s["power_mw"]["mean"] == 6300.0 and s["gpu_pct"]["max"] == 84.0
    assert s["ram_mb"]["max"] == 2501.0 and s["ram_total_mb"] == 3956.0
    assert s["cpu_pct"]["mean"] == 32.2 and s["gpu_temp"]["max"] == 37.0
    assert s["fps_per_watt"] == round(25 / 6.3, 2) and s["mj_per_frame"] == 252.0


def test_orin_format():
    s = summarise(parse(ORIN))
    assert s["power_mw"]["mean"] == 7012.0 and s["gpu_pct"]["mean"] == 60.0
