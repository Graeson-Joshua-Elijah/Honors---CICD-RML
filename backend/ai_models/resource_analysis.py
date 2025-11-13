import psutil
import time
from collections import deque

# keep last N metrics (rolling window)
history = deque(maxlen=50)

def collect_metrics():
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net = psutil.net_io_counters()

    metrics = {
        "timestamp": time.time(),
        "cpu_percent": cpu_percent,
        "mem_percent": mem.percent,
        "mem_used_gb": mem.used / (1024**3),
        "disk_percent": disk.percent,
        "disk_used_gb": disk.used / (1024**3),
        "net_sent_mb": net.bytes_sent / (1024**2),
        "net_recv_mb": net.bytes_recv / (1024**2)
    }

    history.append(metrics)
    return metrics


def analyze_resources():
    """Analyzes recent system metrics and provides AI-based optimization hints."""
    if len(history) < 5:
        collect_metrics()
        return {"status": "collecting", "message": "Collecting more samples for trend analysis"}

    avg_cpu = sum(m["cpu_percent"] for m in history) / len(history)
    avg_mem = sum(m["mem_percent"] for m in history) / len(history)
    avg_disk = sum(m["disk_percent"] for m in history) / len(history)

    recommendations = []

    # CPU Analysis
    if avg_cpu > 80:
        recommendations.append("⚠️ High CPU usage detected. Consider scaling runners or optimizing build scripts.")
    elif avg_cpu < 20:
        recommendations.append("✅ Low CPU utilization. You can reduce parallel jobs to save resources.")

    # Memory Analysis
    if avg_mem > 75:
        recommendations.append("⚠️ High memory usage. Try splitting build stages or increasing available RAM.")
    elif avg_mem < 30:
        recommendations.append("✅ Memory usage is low. You might lower memory allocation for efficiency.")

    # Disk Analysis
    if avg_disk > 85:
        recommendations.append("⚠️ Disk space almost full. Clean old logs, images, or temporary build files.")
    elif avg_disk < 40:
        recommendations.append("✅ Disk usage is healthy.")

    # Basic cost estimate (for simulation)
    estimated_cost = round((avg_cpu / 100) * 0.05 + (avg_mem / 100) * 0.03, 5)

    return {
        "avg_cpu": avg_cpu,
        "avg_mem": avg_mem,
        "avg_disk": avg_disk,
        "estimated_cost_usd": estimated_cost,
        "recommendations": recommendations
    }
