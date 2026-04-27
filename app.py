import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ================== ML PREDICTOR ==================
class MLPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.history = []
        self.is_trained = False

    def add(self, x, y):
        self.history.append([x, y])
        if len(self.history) >= 5:
            self.train()

    def train(self):
        X = np.array(self.history)[:, 0].reshape(-1, 1)
        y = np.array(self.history)[:, 1]
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, x):
        if not self.is_trained:
            return x
        return float(self.model.predict([[x]])[0])


# ================== PROCESS ==================
class Process:
    def __init__(self, pid, at, bt):
        self.pid = pid
        self.at = at
        self.bt = bt


# ================== SCHEDULER ==================
class Scheduler:
    def __init__(self, ml):
        self.ml = ml

    def run(self, processes, algo="FCFS", tq=2):
        procs = [
            {"pid": p.pid, "at": p.at, "bt": p.bt, "rt": p.bt, "done": False}
            for p in processes
        ]

        time = 0
        ready = []
        chart = []
        current = None
        quantum = tq

        max_time = 1000  # safety limit

        while time < max_time:
            # Add processes
            for p in procs:
                if p["at"] == time and not p["done"]:
                    ready.append(p)

            # Select process
            if not current and ready:
                if algo == "FCFS":
                    current = ready.pop(0)

                elif algo == "SJF":
                    ready.sort(key=lambda x: x["bt"])
                    current = ready.pop(0)

                elif algo == "AI-SJF":
                    ready.sort(key=lambda x: self.ml.predict(x["bt"]))
                    current = ready.pop(0)

                elif algo == "RR":
                    current = ready.pop(0)
                    quantum = tq

            if current:
                current["rt"] -= 1
                time += 1

                if algo == "RR":
                    quantum -= 1

                # Finished
                if current["rt"] == 0:
                    current["done"] = True
                    tat = time - current["at"]
                    wt = tat - current["bt"]

                    chart.append((time - current["bt"], time, current["pid"]))

                    self.ml.add(current["bt"], current["bt"])

                    current["tat"] = tat
                    current["wt"] = wt
                    current = None

                # RR quantum over
                elif algo == "RR" and quantum == 0:
                    ready.append(current)
                    current = None

            else:
                time += 1

            if all(p["done"] for p in procs):
                break

        avg_wt = sum(p["wt"] for p in procs) / len(procs)
        avg_tat = sum(p["tat"] for p in procs) / len(procs)

        return avg_wt, avg_tat, chart, procs


# ================== UI ==================
st.set_page_config(layout="wide")
st.title("🤖 AI CPU Scheduler")

if "processes" not in st.session_state:
    st.session_state.processes = []
    st.session_state.ml = MLPredictor()

scheduler = Scheduler(st.session_state.ml)

# Input
st.header("➕ Add Process")

col1, col2, col3 = st.columns(3)
pid = col1.number_input("Process ID", step=1)
at = col2.number_input("Arrival Time", step=1)
bt = col3.number_input("Burst Time", step=1)

if st.button("Add Process"):
    st.session_state.processes.append(Process(pid, at, bt))

# Display
if st.session_state.processes:
    st.subheader("📋 Process Table")
    df = pd.DataFrame([vars(p) for p in st.session_state.processes])
    st.dataframe(df)

    # Comparison
    st.header("📊 Algorithm Comparison")

    algorithms = ["FCFS", "SJF", "AI-SJF", "RR"]
    tq = st.slider("Time Quantum (RR)", 1, 10, 2)

    if st.button("Run Comparison"):
        results = []

        for algo in algorithms:
            avg_wt, avg_tat, chart, _ = scheduler.run(
                st.session_state.processes, algo, tq
            )

            results.append({
                "Algorithm": algo,
                "Avg Waiting Time": avg_wt,
                "Avg Turnaround Time": avg_tat
            })

        result_df = pd.DataFrame(results)

        st.subheader("📈 Performance Table")
        st.dataframe(result_df)

        fig = px.bar(
            result_df,
            x="Algorithm",
            y=["Avg Waiting Time", "Avg Turnaround Time"],
            barmode="group"
        )
        st.plotly_chart(fig)

    # Gantt Chart
    st.header("📉 Gantt Chart")

    algo_choice = st.selectbox("Select Algorithm", algorithms)

    if st.button("Show Gantt Chart"):
        avg_wt, avg_tat, chart, _ = scheduler.run(
            st.session_state.processes, algo_choice, tq
        )

        gantt_df = pd.DataFrame(chart, columns=["Start", "End", "PID"])

        fig = px.timeline(
            gantt_df,
            x_start="Start",
            x_end="End",
            y="PID"
        )
        fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig)