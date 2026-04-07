from flask import Flask, render_template, request 
import pandas as pd
import plotly.express as px
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image
from flask import session

app = Flask(__name__)
app.secret_key = "supersecretkey"

def generate_insights(data, col):
    insights = []

    values = data[col].dropna()

    mean = values.mean()
    median = values.median()
    std = values.std()

    # --------------------------
    # Basic Stats
    # --------------------------
    insights.append(f"The average value of {col} is {round(mean,2)}.")
    insights.append(f"The median is {round(median,2)}.")

    # Distribution
    if mean > median:
        insights.append("The data appears slightly right-skewed.")
    elif mean < median:
        insights.append("The data appears slightly left-skewed.")
    else:
        insights.append("The data is symmetrically distributed.")

    # Variability
    if std > mean * 0.5:
        insights.append("There is high variation in the data.")
    else:
        insights.append("The data is relatively stable.")

    # Trend Detection
    trend = values.diff().mean()

    if trend > 0:
        insights.append("Overall trend shows an increasing pattern.")
    elif trend < 0:
        insights.append("Overall trend shows a decreasing pattern.")
    else:
        insights.append("No clear trend detected.")

    # --------------------------
    # Anomaly Detection
    # --------------------------
    upper = mean + 2 * std
    lower = mean - 2 * std

    outliers = values[(values > upper) | (values < lower)]

    if len(outliers) > 0:
        insights.append(f"{len(outliers)} potential outliers detected.")
    else:
        insights.append("No significant outliers detected.")

    # Range Info
    insights.append(f"Maximum value is {values.max()} and minimum value is {values.min()}.")
    return insights


def generate_narrative(data, col):
    values = data[col].dropna()

    mean = round(values.mean(), 2)
    trend = values.diff().mean()

    if trend > 0:
        trend_text = "an overall increasing trend"
    elif trend < 0:
        trend_text = "a decreasing trend"
    else:
        trend_text = "no clear trend"

    explanation = (
        f"The dataset for '{col}' shows an average value of {mean}. "
        f"From the observed values, there is {trend_text}. "
        f"The variation in the data suggests that the values may fluctuate, "
        f"and some extreme values could exist. Overall, this feature provides "
        f"important insights into the dataset behavior."
    )

    return explanation


def generate_correlation(data, col):
    numeric_data = data.select_dtypes(include='number')

    if len(numeric_data.columns) < 2:
        return []

    correlations = numeric_data.corr()[col].drop(col)

    result = []

    for other_col, value in correlations.items():
        if abs(value) > 0.7:
            strength = "strong"
        elif abs(value) > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        direction = "positive" if value > 0 else "negative"

        result.append(f"{col} has a {strength} {direction} correlation with {other_col} ({round(value,2)}).")

    return result

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    else:
        return obj

@app.route('/', methods=['GET', 'POST'])
def home():
    columns = []
    summary = None
    graphs = None
    selected_col = None   # NEW
    insights = None
    narrative = None
    correlation = None
    file_path = None

    if request.method == 'POST':
        file = request.files['file']

        if file and file.filename != "":
            # ✅ ADD HERE
            if not os.path.exists("uploads"):
                os.makedirs("uploads")

            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            session['file_path'] = file_path
            data = pd.read_csv(file_path)

            columns = list(data.columns)
            numeric_cols = data.select_dtypes(include='number').columns
            selected_col = request.form.get('column')

            if selected_col not in numeric_cols:
                selected_col = numeric_cols[0]
            min_val = request.form.get('min')
            max_val = request.form.get('max')

            if min_val:
                data = data[data[selected_col] >= float(min_val)]
            if max_val:
                data = data[data[selected_col] <= float(max_val)]
            summary = {
                "Total": float(data[selected_col].sum()),
                "Average": round(float(data[selected_col].mean()), 2),
                "Max": float(data[selected_col].max()),
                "Min": float(data[selected_col].min())
            }
            data = data.reset_index().rename(columns={'index': 'Index'})

            # ✅ Plotly graphs
            fig1 = px.line(data, x='Index', y=selected_col, title="Line Chart")
            fig2 = px.bar(data, x='Index', y=selected_col, title="Bar Chart")

            vc = data[selected_col].value_counts().reset_index()
            vc.columns = ['Value', 'Count']
            fig3 = px.pie(vc, names='Value', values='Count', title="Pie Chart")

            fig4 = px.histogram(data, x=selected_col, title="Histogram")

            graphs = {
                "line": convert_ndarray(fig1.to_dict()),
                "bar": convert_ndarray(fig2.to_dict()),
                "pie": convert_ndarray(fig3.to_dict()),
                "hist": convert_ndarray(fig4.to_dict())
            }

            # ✅ Matplotlib graphs
            if not os.path.exists('static'):
                os.makedirs('static')

            plt.figure()
            data[selected_col].plot(title="Line Chart")
            plt.savefig('static/line.png')
            plt.close()

            plt.figure()
            data[selected_col].value_counts().head(10).plot(kind='bar', title="Bar Chart")
            plt.savefig('static/bar.png')
            plt.close()

            plt.figure()
            data[selected_col].value_counts().head(5).plot(kind='pie', autopct='%1.1f%%', title="Pie Chart")
            plt.savefig('static/pie.png')
            plt.close()

            plt.figure()
            data[selected_col].plot(kind='hist', title="Histogram")
            plt.savefig('static/hist.png')
            plt.close()
            
            insights = generate_insights(data, selected_col)
            narrative = generate_narrative(data, selected_col)
            correlation = generate_correlation(data, selected_col)

    return render_template(
        "index.html",
        columns=columns,
        summary=summary,
        graphs=graphs,
        selected_col=selected_col,
        insights=insights,
        narrative=narrative,
        correlation=correlation,
        file_path=file_path
    )
    
@app.route('/download_pdf', methods=['POST'])
def download_pdf():

    file_path = session.get('file_path')

    if not file_path or not os.path.exists(file_path):
        return "Session not found. Please upload again."
    
    if not os.path.exists(file_path):
        return f"File not found at {file_path}"

    data = pd.read_csv(file_path)

    selected_col = request.form.get('column')
    numeric_cols = data.select_dtypes(include='number').columns

    if selected_col not in numeric_cols:
        selected_col = numeric_cols[0]

    # Summary
    summary = {
        "Total": float(data[selected_col].sum()),
        "Average": round(float(data[selected_col].mean()), 2),
        "Max": float(data[selected_col].max()),
        "Min": float(data[selected_col].min())
    }

    insights = generate_insights(data, selected_col)
    narrative = generate_narrative(data, selected_col)
    correlation = generate_correlation(data, selected_col)

    # PDF FILE
    pdf_path = f"report_{selected_col}.pdf"

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    content = []

    # TITLE
    content.append(Paragraph("📊 AutoInsight Report", styles['Title']))
    content.append(Spacer(1, 15))

    content.append(Paragraph(f"<b>Selected Column:</b> {selected_col}", styles['Normal']))
    content.append(Spacer(1, 10))

    # SUMMARY
    content.append(Paragraph("📈 Summary", styles['Heading2']))
    for k, v in summary.items():
        content.append(Paragraph(f"{k}: {v}", styles['Normal']))

    content.append(Spacer(1, 15))

    # AI EXPLANATION
    content.append(Paragraph("🤖 AI Explanation", styles['Heading2']))
    content.append(Paragraph(narrative, styles['Normal']))
    content.append(Spacer(1, 15))

    # INSIGHTS
    content.append(Paragraph("🧠 Insights", styles['Heading2']))
    for i in insights:
        content.append(Paragraph(f"• {i}", styles['Normal']))

    content.append(Spacer(1, 15))

    # CORRELATION
    if correlation:
        content.append(Paragraph("🔗 Correlation", styles['Heading2']))
        for c in correlation:
            content.append(Paragraph(f"• {c}", styles['Normal']))
        content.append(Spacer(1, 15))

    # GRAPHS (IMPORTANT)
    content.append(Paragraph("📊 Graphs", styles['Heading2']))
    content.append(Spacer(1, 10))

    graph_files = [
        ("Line Chart", "static/line.png"),
        ("Bar Chart", "static/bar.png"),
        ("Pie Chart", "static/pie.png"),
        ("Histogram", "static/hist.png")
    ]

    for title, path in graph_files:
        if os.path.exists(path):
            content.append(Paragraph(title, styles['Heading3']))
            content.append(Spacer(1, 5))

            img = Image(path, width=400, height=250)
            content.append(img)
            content.append(Spacer(1, 15))

    # BUILD PDF
    doc.build(content)

    from flask import send_file
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
