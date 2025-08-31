# Monte Carlo Risk Simulation (GPU with Nebius)

This project shows how to run a **Monte Carlo simulation** to estimate portfolio risk using millions of random scenarios. It runs super fast on a **Nebius AI Cloud H100 GPU VM** (with [CuPy](https://cupy.dev/)) but will also work on CPU if you don’t have a GPU.

---

## 🚀 Quick Start

1. **SSH into your Nebius VM**
   ```bash
   ssh <USER>@<PUBLIC_IP>
````

2. **Set up Python**

   ```bash
   python3 -m venv mcenv
   source mcenv/bin/activate
   pip install --upgrade pip
   pip install cupy-cuda12x numpy pandas tabulate
   ```

3. **Copy the script**

   ```bash
   scp -i ~/.ssh/id_rsa main.py <USER>@<PUBLIC_IP>:/home/<USER>/
   ```

4. **Run the simulation**

   ```bash
   # quick test
   N_PATHS=100000 HORIZON_YEARS=0.5 python main.py

   # full run (10 million scenarios)
   N_PATHS=10000000 HORIZON_YEARS=1 python main.py
   ```

---

## 📊 What You Get

* Mean and standard deviation of portfolio value
* Value at Risk (VaR 95%) and Conditional VaR (CVaR 95%)
* Probability of losing money

Example:

```
Device: GPU (CuPy/CUDA)
Scenarios: 10,000,000
VaR(95%): $15,433
CVaR(95%): $19,201
P(Loss): 27.8%
```

---

## ✅ Notes

* File name: `main.py`
* Works best on Nebius H100 GPUs (CUDA 12.x)
* Stop or delete your VM after use to avoid charges
