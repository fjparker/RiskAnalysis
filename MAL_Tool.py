import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import io
from datetime import datetime

# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================

def required_return(present_value, future_value, years, contributions, contribution_growth):
    """Calculate the required return to meet the goal"""
    if contribution_growth == 0:
        FV_annuity = contributions * years
    else:
        FV_annuity = contributions * ((1 + contribution_growth) ** years) / contribution_growth
    return ((future_value - FV_annuity) / present_value) ** (1/years) - 1

def recovery_return(allocation_to_stocks, allocation_to_bonds, allocation_to_cash, recovery_percentile=0.65):
    """Calculate the expected recovery return at a given percentile"""
    covar_table = np.array([0.03960, -0.00046, 0.00000,
                            -0.00046, 0.00609, 0.00071,
                            0.00000, 0.00000, 0.00000]).reshape(3,3)
    return_vector = np.array([0.120, 0.04, 0.03])
    weight_vector = np.array([allocation_to_stocks, allocation_to_bonds, allocation_to_cash])
    
    portfolio_E_return = return_vector @ weight_vector
    portfolio_E_stdev = (np.transpose(weight_vector) @ covar_table @ weight_vector) ** 0.5
    
    return norm.ppf(recovery_percentile, loc=portfolio_E_return, scale=portfolio_E_stdev)

def maximum_allowable_loss(present_value, future_value, contributions, contribution_growth, years, recovery_ret):
    """Calculate the maximum allowable loss in year 1"""
    if contribution_growth == 0:
        FV_annuity = contributions * years
    else:
        FV_annuity = contributions * ((1 + contribution_growth) ** years) / contribution_growth
    
    numerator = future_value - FV_annuity
    denominator = (present_value + contributions) * (1 + recovery_ret) ** (years - 1)
    
    return (numerator / denominator) - 1

def portfolio_expected_return(allocation_to_stocks, allocation_to_bonds, allocation_to_cash):
    """Calculate the expected return of the portfolio"""
    return_vector = np.array([0.120, 0.04, 0.03])
    weight_vector = np.array([allocation_to_stocks, allocation_to_bonds, allocation_to_cash])
    return return_vector @ weight_vector

def portfolio_std_dev(allocation_to_stocks, allocation_to_bonds, allocation_to_cash):
    """Calculate portfolio standard deviation"""
    covar_table = np.array([0.03960, -0.00046, 0.00000,
                            -0.00046, 0.00609, 0.00071,
                            0.00000, 0.00000, 0.00000]).reshape(3,3)
    weight_vector = np.array([allocation_to_stocks, allocation_to_bonds, allocation_to_cash])
    return (np.transpose(weight_vector) @ covar_table @ weight_vector) ** 0.5

def calculate_new_contribution(present_value, future_value, years, contribution_growth, port_expected_return):
    """Calculate new contribution amount needed (Option 1)"""
    if contribution_growth == 0 or abs(port_expected_return - contribution_growth) < 0.0001:
        # Simplified calculation when growth rates are equal or zero
        numerator = future_value - present_value * (1 + port_expected_return) ** years
        denominator = years * (1 + port_expected_return) ** (years - 1)
        return numerator / denominator
    else:
        # Full formula
        numerator = future_value - present_value * (1 + port_expected_return) ** years
        denominator = ((1 + port_expected_return) ** years - (1 + contribution_growth) ** years) / (port_expected_return - contribution_growth)
        return numerator / denominator

def calculate_adjusted_years(present_value, future_value, contributions, contribution_growth, port_expected_return, initial_years):
    """Calculate adjusted years needed (Option 2) - numerical solution"""
    from scipy.optimize import fsolve
    
    def equation(years):
        if contribution_growth == 0:
            FV_annuity = contributions * years
        else:
            FV_annuity = contributions * ((1 + port_expected_return) ** years - (1 + contribution_growth) ** years) / (port_expected_return - contribution_growth)
        FV_calculated = present_value * (1 + port_expected_return) ** years + FV_annuity
        return FV_calculated - future_value
    
    result = fsolve(equation, initial_years + 5)
    return result[0]

def optimize_allocation(required_ret):
    """Find allocation that maximizes probability of exceeding required return (Option 3)"""
    covar_table = np.array([0.03960, -0.00046, 0.00000,
                            -0.00046, 0.00609, 0.00071,
                            0.00000, 0.00000, 0.00000]).reshape(3,3)
    return_vector = np.array([0.120, 0.04, 0.03])
    
    def objective(weights):
        """Negative probability of exceeding required return"""
        portfolio_return = return_vector @ weights
        portfolio_std = (weights @ covar_table @ weights) ** 0.5
        prob_exceed = 1 - norm.cdf(required_ret, loc=portfolio_return, scale=portfolio_std)
        return -prob_exceed  # Minimize negative = maximize positive
    
    # Constraints: weights sum to 1, all non-negative
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1), (0, 1), (0, 1)]
    initial_guess = [0.33, 0.34, 0.33]
    
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x  # Returns [stocks, bonds, cash]

# ============================================================================
# CHART GENERATION FUNCTIONS
# ============================================================================

def create_effects_of_losses_chart(present_value, future_value, contributions, contribution_growth, years, port_expected_ret):
    """Create the Effects of Losses chart (Chunk 5)"""
    losses = np.arange(-0.40, 0.41, 0.05)
    recovery_requirements = []
    
    for loss in losses:
        new_pv = present_value * (1 + loss)
        remaining_years = years - 1
        if remaining_years > 0:
            req_return = required_return(new_pv, future_value, remaining_years, contributions, contribution_growth)
        else:
            req_return = 0
        recovery_requirements.append(req_return)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses * 100, np.array(recovery_requirements) * 100, linewidth=2, color='#2E5090')
    ax.axhline(y=port_expected_ret * 100, color='red', linestyle='--', linewidth=1.5, label='Portfolio Expected Return')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Hypothetical Loss (%)', fontsize=11)
    ax.set_ylabel('Recovery Return Requirement (%)', fontsize=11)
    ax.set_title('Effects of Losses', fontsize=13, fontweight='bold')
    ax.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def create_allocation_chart(stocks, bonds, cash, title="Your Allocation"):
    """Create horizontal bar chart for allocation"""
    fig, ax = plt.subplots(figsize=(8, 3))
    
    categories = ['Cash', 'Bonds', 'Stocks']
    values = [cash * 100, bonds * 100, stocks * 100]
    colors_list = ['#90EE90', '#FFD700', '#4169E1']
    
    ax.barh(categories, values, color=colors_list)
    ax.set_xlabel('Allocation (%)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(0, 100)
    
    for i, v in enumerate(values):
        ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def create_comparison_chart(current_stocks, current_bonds, current_cash, 
                           suggested_stocks, suggested_bonds, suggested_cash):
    """Create comparison bar chart for adjusted allocation"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    categories = ['Stocks', 'Bonds', 'Cash']
    current = [current_stocks * 100, current_bonds * 100, current_cash * 100]
    suggested = [suggested_stocks * 100, suggested_bonds * 100, suggested_cash * 100]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.barh(x - width/2, suggested, width, label='Suggested Allocation', color='#90EE90')
    ax.barh(x + width/2, current, width, label='Current Allocation', color='#4169E1')
    
    ax.set_yticks(x)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Allocation (%)', fontsize=11)
    ax.set_title('Adjusted Allocation', fontsize=13, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 100)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def create_return_gap_chart(required_ret, portfolio_ret):
    """Create return gap analysis bar chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    categories = ['Portfolio Expected\nReturn', 'Required Return']
    values = [portfolio_ret * 100, required_ret * 100]
    colors_list = ['#4169E1', '#DC143C']
    
    ax.bar(categories, values, color=colors_list, width=0.5)
    ax.set_ylabel('Return (%)', fontsize=11)
    ax.set_title('Return Gap Analysis', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(values):
        ax.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def create_comparison_bar_chart(value1, value2, label1, label2, title, ylabel):
    """Generic comparison bar chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    categories = [label1, label2]
    values = [value1 * 100, value2 * 100]
    colors_list = ['#4169E1', '#90EE90']
    
    ax.bar(categories, values, color=colors_list, width=0.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(values):
        ax.text(i, v + (max(values) * 0.02), f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

# ============================================================================
# PDF GENERATION
# ============================================================================

def generate_pdf(client_name, inputs, calculations):
    """Generate the complete PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E5090'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2E5090'),
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    
    metric_style = ParagraphStyle(
        'MetricStyle',
        parent=styles['Normal'],
        fontSize=28,
        textColor=colors.HexColor('#DC143C'),
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    description_style = ParagraphStyle(
        'DescriptionStyle',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_LEFT,
        spaceAfter=10
    )
    
    # PAGE 1 - Cover Page
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"HOW MUCH IS TOO MUCH Analysis", title_style))
    story.append(Paragraph(f"{client_name}", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Prepared on {datetime.now().strftime('%m/%d/%Y')}", styles['Normal']))
    story.append(PageBreak())
    
    # PAGE 2 - Core Metrics
    story.append(Paragraph("HOW MUCH IS TOO MUCH ANALYSIS", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Three metrics in a row
    metrics_data = [
        [
            Paragraph(f"{calculations['required_return']*100:.2f}%", metric_style),
            Paragraph(f"{calculations['max_loss']*100:.2f}%", metric_style),
            Paragraph(f"{calculations['recovery_return']*100:.2f}%", metric_style)
        ],
        [
            Paragraph("<b>Required Return</b><br/>Given your scenario, this is the average return which is required to attain your goal.", description_style),
            Paragraph("<b>Portfolio Loss Tolerance</b><br/>Given your scenario, this is the worst return your portfolio can sustain next year before derailing your plan.", description_style),
            Paragraph("<b>Recovery Return</b><br/>With your current allocation, we can reasonably expect it to recover at this rate after a downturn.", description_style)
        ]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch, 2.5*inch])
    metrics_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.5*inch))
    
    # Effects of Losses Chart
    story.append(Paragraph("Effects of Losses", heading_style))
    story.append(Paragraph("This chart shows the effects of various losses on the recovery requirement.", description_style))
    chart_buf = create_effects_of_losses_chart(
        inputs['present_value'], inputs['future_value'], inputs['contributions'],
        inputs['contribution_growth'], inputs['years'], calculations['portfolio_expected_return']
    )
    story.append(Image(chart_buf, width=6*inch, height=3.6*inch))
    story.append(PageBreak())
    
    # PAGE 3 - Portfolio Analysis
    story.append(Paragraph("HOW MUCH IS TOO MUCH ANALYSIS", title_style))
    
    # Your Allocation
    story.append(Paragraph("Your Allocation", heading_style))
    story.append(Paragraph("This is your current portfolio allocation.", description_style))
    alloc_buf = create_allocation_chart(inputs['stocks'], inputs['bonds'], inputs['cash'])
    story.append(Image(alloc_buf, width=5*inch, height=2*inch))
    story.append(Spacer(1, 0.3*inch))
    
    # Maximum Losses Table
    story.append(Paragraph("Maximum Losses", heading_style))
    losses_data = [
        ['Asset Class', 'Maximum Losses Expected', 'Maximum Losses Suggested'],
        ['Stocks', '-43.84%', f"{calculations['stocks_suggested_loss']*100:.2f}%"],
        ['Bonds', '-11.12%', '-5.00%'],
        ['Cash', '0.03%', '0.00%'],
        ['Portfolio', '0.00%', f"{calculations['max_loss']*100:.2f}%"]
    ]
    losses_table = Table(losses_data, colWidths=[2*inch, 2*inch, 2*inch])
    losses_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E5090')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    story.append(losses_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Strategies for Mitigating Losses</b><br/>Stop-losses and hedging strategies carry their own unique risks and fees, but they may be appropriate in helping prevent unrecoverable losses in your portfolio.", description_style))
    story.append(PageBreak())
    
    # PAGE 4 - Gap Analysis & Options
    story.append(Paragraph("HOW MUCH IS TOO MUCH ANALYSIS", title_style))
    
    # Return Gap Analysis
    story.append(Paragraph("Return Gap Analysis", heading_style))
    meets_requirements = calculations['portfolio_expected_return'] >= calculations['required_return']
    status_text = "DOES" if meets_requirements else "DOES NOT"
    story.append(Paragraph(f"Your current portfolio <b>{status_text}</b> meet your minimum return requirements.", description_style))
    
    if not meets_requirements:
        gap = calculations['required_return'] - calculations['portfolio_expected_return']
        story.append(Paragraph(f"To meet your stated needs, you need an additional <b>{gap*100:.2f}%</b> per year in portfolio return.", description_style))
    
    gap_buf = create_return_gap_chart(calculations['required_return'], calculations['portfolio_expected_return'])
    story.append(Image(gap_buf, width=5*inch, height=3*inch))
    story.append(Spacer(1, 0.3*inch))
    
    # Options for Meeting Your Needs
    story.append(Paragraph("Options for Meeting Your Needs", heading_style))
    
    story.append(Paragraph(f"<b>OPTION 1: Change Your Savings Pattern</b><br/>You could adjust your contributions to <b>${calculations['new_contribution']:,.2f}</b> per year to help fund your goal.", description_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph(f"<b>OPTION 2: Adjust Goal Date</b><br/>If you adjust your goal by <b>{calculations['years_adjustment']:.1f}</b> years, you can still attain your funding requirement.", description_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph(f"<b>OPTION 3: Reallocate</b><br/>You could consider reallocating your portfolio. An allocation of Stocks <b>{calculations['suggested_stocks']*100:.2f}%</b>, Bonds <b>{calculations['suggested_bonds']*100:.2f}%</b>, Cash <b>{calculations['suggested_cash']*100:.2f}%</b> could help you reach your funding requirement.", description_style))
    story.append(PageBreak())
    
    # PAGE 5 - Adjusted Allocation Scenario
    story.append(Paragraph("HOW MUCH IS TOO MUCH ANALYSIS", title_style))
    story.append(Paragraph("PURSUING AN ADJUSTED ALLOCATION", heading_style))
    
    comp_buf = create_comparison_chart(
        inputs['stocks'], inputs['bonds'], inputs['cash'],
        calculations['suggested_stocks'], calculations['suggested_bonds'], calculations['suggested_cash']
    )
    story.append(Image(comp_buf, width=5*inch, height=3*inch))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph(f"<b>Portfolio Loss Tolerance after Adjustment</b><br/>{calculations['adjusted_max_loss']*100:.2f}%", metric_style))
    story.append(Paragraph("By making the suggested changes, your portfolio should <b>TOLERATE FEWER LOSSES</b>. Under the adjusted allocation, this is the worst return your portfolio can sustain next year before derailing your plan.", description_style))
    story.append(PageBreak())
    
    # PAGE 6 - Adjusted Recovery Analysis
    story.append(Paragraph("HOW MUCH IS TOO MUCH ANALYSIS", title_style))
    
    story.append(Paragraph(f"<b>New Recovery Return</b><br/>{calculations['adjusted_recovery_return']*100:.2f}%", metric_style))
    story.append(Paragraph("With your adjusted allocation, we can reasonably expect it to recover at this rate after a downturn.", description_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Adjusted Maximum Losses Table
    adj_losses_data = [
        ['Asset Class', 'Maximum Losses Expected', 'Maximum Losses Suggested'],
        ['Stocks', '-43.84%', f"{calculations['adjusted_stocks_suggested_loss']*100:.2f}%"],
        ['Bonds', '-11.12%', '-5.00%'],
        ['Cash', '0.03%', '0.00%'],
        ['Portfolio', '0.00%', f"{calculations['adjusted_max_loss']*100:.2f}%"]
    ]
    adj_losses_table = Table(adj_losses_data, colWidths=[2*inch, 2*inch, 2*inch])
    adj_losses_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E5090')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    story.append(adj_losses_table)
    story.append(Spacer(1, 0.3*inch))
    
    recovery_comp_buf = create_comparison_bar_chart(
        calculations['recovery_return'], calculations['adjusted_recovery_return'],
        'Current Recovery Return', 'Recovery Return After Adjustment',
        'Recovery Return Comparison', 'Return (%)'
    )
    story.append(Image(recovery_comp_buf, width=5*inch, height=3*inch))
    story.append(PageBreak())
    
    # PAGE 7 - Delayed Goal Scenario
    story.append(Paragraph("HOW MUCH IS TOO MUCH ANALYSIS", title_style))
    story.append(Paragraph("DELAYING YOUR GOAL", heading_style))
    
    story.append(Paragraph(f"<b>Adjusted Required Return</b><br/>{calculations['delayed_required_return']*100:.2f}%", metric_style))
    story.append(Paragraph("By delaying your goal, this is the average return which is required.", description_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph(f"<b>Adjusted Portfolio Loss Tolerance</b><br/>{calculations['delayed_max_loss']*100:.2f}%", metric_style))
    story.append(Paragraph("By delaying your goal, your portfolio should <b>BETTER TOLERATE LOSSES</b>. If you delay your goal, this is the worst return your portfolio can sustain next year before derailing your plan.", description_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Delayed Maximum Losses Table
    delayed_losses_data = [
        ['Asset Class', 'Maximum Losses Expected', 'Maximum Losses Suggested'],
        ['Stocks', '-43.84%', f"{calculations['delayed_stocks_suggested_loss']*100:.2f}%"],
        ['Bonds', '-11.12%', '-5.00%'],
        ['Cash', '0.03%', '0.00%'],
        ['Portfolio', '0.00%', f"{calculations['delayed_max_loss']*100:.2f}%"]
    ]
    delayed_losses_table = Table(delayed_losses_data, colWidths=[2*inch, 2*inch, 2*inch])
    delayed_losses_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E5090')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    story.append(delayed_losses_table)
    story.append(Spacer(1, 0.3*inch))
    
    loss_comp_buf = create_comparison_bar_chart(
        calculations['max_loss'], calculations['delayed_max_loss'],
        'Current Maximum Loss Tolerance', 'Maximum Loss Tolerance by Delaying Goal',
        'Maximum Loss Tolerance Comparison', 'Loss Tolerance (%)'
    )
    story.append(Image(loss_comp_buf, width=5*inch, height=3*inch))
    story.append(PageBreak())
    
    # PAGES 8-10 - Disclosures (simplified version)
    story.append(Paragraph("IMPORTANT DISCLOSURES", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    disclosures = [
        ("KEY ASSUMPTIONS:", "A key assumption in this tool is the mean expected return of each asset class. To calculate the mean expected return for an asset class the historical returns of the asset class are averaged."),
        ("", "This document and any other materials provided to you are intended only for discussion purposes and are not intended as an offer to sell or solicitation of an offer to buy with respect to the purchase or sale of any security and should not be relied upon by you in evaluating the merits of investing in any securities."),
        ("", "The projections and other information you will see here about the likelihood of various outcomes are hypothetical in nature, do not reflect actual investment results, and are not guarantees of future results. Past performance results relate only to the time periods indicated and are not an indication of, nor a reliable proxy for future performance."),
        ("", "The output of this HOW MUCH IS TOO MUCH calculator may vary with each use and over time. The tool does not consider specific securities held by you and does not therefore attempt to predict the future value of any specific securities."),
        ("", "No proprietary technology or asset allocation model is a guarantee against loss of principal. There can be no assurance that an investment strategy based on the HMITM analysis tool will be successful."),
        ("RISK CONSIDERATIONS:", "Asset Class Risk. Each of these asset classes has its own set of investment characteristics and risks and investors should consider these risks carefully prior to making any investments. Credit Risk, Currency Risk, Debt Securities Risk, Derivatives Risk, Dividend-Paying Stock Risk, Foreign and Emerging Markets Risk, Growth Investing Risk, Hedging Risk, High Yield Securities Risk, Interest Rate Risk, Leverage Risk, and other risks apply."),
    ]
    
    for heading, text in disclosures:
        if heading:
            story.append(Paragraph(f"<b>{heading}</b>", styles['Normal']))
        story.append(Paragraph(text, description_style))
        story.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="Risk Analysis Report Generator", layout="wide")

st.title("HOW MUCH IS TOO MUCH - Risk Analysis Report Generator")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("Client Information")
client_name = st.sidebar.text_input("Client Name", "Sample Client")

st.sidebar.header("Portfolio Details")
present_value = st.sidebar.number_input("Current Portfolio Value ($)", min_value=0.0, value=100000.0, step=1000.0)
future_value = st.sidebar.number_input("Target Value ($)", min_value=0.0, value=500000.0, step=1000.0)
years = st.sidebar.number_input("Time Horizon (years)", min_value=1, value=10, step=1)
contributions = st.sidebar.number_input("Annual Contributions ($)", min_value=0.0, value=10000.0, step=500.0)
contribution_growth = st.sidebar.number_input("Contribution Growth Rate", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.3f")

st.sidebar.header("Current Allocation")
stocks = st.sidebar.slider("Stocks (%)", 0, 100, 60) / 100
bonds = st.sidebar.slider("Bonds (%)", 0, 100, 35) / 100
cash = st.sidebar.slider("Cash (%)", 0, 100, 5) / 100

# Validate allocation sums to 100%
total_allocation = stocks + bonds + cash
if abs(total_allocation - 1.0) > 0.01:
    st.sidebar.error(f"⚠️ Allocation must sum to 100% (currently {total_allocation*100:.1f}%)")

st.sidebar.header("Advanced Settings")
recovery_percentile = st.sidebar.slider("Recovery Percentile", 0.0, 1.0, 0.65, 0.05)

# Main content
if st.sidebar.button("Generate Report", type="primary", use_container_width=True):
    if abs(total_allocation - 1.0) > 0.01:
        st.error("Please ensure portfolio allocation sums to 100%")
    else:
        with st.spinner("Generating risk analysis report..."):
            # Store inputs
            inputs = {
                'present_value': present_value,
                'future_value': future_value,
                'years': years,
                'contributions': contributions,
                'contribution_growth': contribution_growth,
                'stocks': stocks,
                'bonds': bonds,
                'cash': cash,
                'recovery_percentile': recovery_percentile
            }
            
            # Calculate all metrics
            calculations = {}
            
            # Core metrics
            calculations['required_return'] = required_return(present_value, future_value, years, contributions, contribution_growth)
            calculations['recovery_return'] = recovery_return(stocks, bonds, cash, recovery_percentile)
            calculations['max_loss'] = maximum_allowable_loss(present_value, future_value, contributions, contribution_growth, years, calculations['recovery_return'])
            calculations['portfolio_expected_return'] = portfolio_expected_return(stocks, bonds, cash)
            
            # Stocks suggested loss
            if stocks > 0:
                calculations['stocks_suggested_loss'] = (bonds * (-0.05) - calculations['max_loss']) / (-stocks)
            else:
                calculations['stocks_suggested_loss'] = 0
            
            # Options
            calculations['new_contribution'] = calculate_new_contribution(present_value, future_value, years, contribution_growth, calculations['portfolio_expected_return'])
            calculations['adjusted_years'] = calculate_adjusted_years(present_value, future_value, contributions, contribution_growth, calculations['portfolio_expected_return'], years)
            calculations['years_adjustment'] = calculations['adjusted_years'] - years
            
            # Option 3 - Optimized allocation
            suggested_allocation = optimize_allocation(calculations['required_return'])
            calculations['suggested_stocks'] = suggested_allocation[0]
            calculations['suggested_bonds'] = suggested_allocation[1]
            calculations['suggested_cash'] = suggested_allocation[2]
            
            # Adjusted allocation metrics
            calculations['adjusted_recovery_return'] = recovery_return(
                calculations['suggested_stocks'], 
                calculations['suggested_bonds'], 
                calculations['suggested_cash'], 
                recovery_percentile
            )
            calculations['adjusted_max_loss'] = maximum_allowable_loss(
                present_value, future_value, contributions, contribution_growth, years, 
                calculations['adjusted_recovery_return']
            )
            
            # Adjusted stocks suggested loss
            if calculations['suggested_stocks'] > 0:
                calculations['adjusted_stocks_suggested_loss'] = (
                    calculations['suggested_bonds'] * (-0.05) - calculations['adjusted_max_loss']
                ) / (-calculations['suggested_stocks'])
            else:
                calculations['adjusted_stocks_suggested_loss'] = 0
            
            # Delayed goal metrics
            calculations['delayed_required_return'] = required_return(
                present_value, future_value, calculations['adjusted_years'], 
                contributions, contribution_growth
            )
            calculations['delayed_recovery_return'] = recovery_return(stocks, bonds, cash, recovery_percentile)
            calculations['delayed_max_loss'] = maximum_allowable_loss(
                present_value, future_value, contributions, contribution_growth, 
                calculations['adjusted_years'], calculations['delayed_recovery_return']
            )
            
            # Delayed stocks suggested loss
            if stocks > 0:
                calculations['delayed_stocks_suggested_loss'] = (
                    bonds * (-0.05) - calculations['delayed_max_loss']
                ) / (-stocks)
            else:
                calculations['delayed_stocks_suggested_loss'] = 0
            
            # Display key results
            st.success("✅ Analysis Complete!")
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Required Return", f"{calculations['required_return']*100:.2f}%")
                st.caption("Average return needed to reach goal")
            
            with col2:
                st.metric("Portfolio Loss Tolerance", f"{calculations['max_loss']*100:.2f}%")
                st.caption("Worst Year 1 return before derailing plan")
            
            with col3:
                st.metric("Recovery Return", f"{calculations['recovery_return']*100:.2f}%")
                st.caption("Expected recovery rate after downturn")
            
            st.markdown("---")
            
            # Gap Analysis
            st.subheader("Return Gap Analysis")
            meets_requirements = calculations['portfolio_expected_return'] >= calculations['required_return']
            
            if meets_requirements:
                st.success(f"✅ Your portfolio **DOES** meet your minimum return requirements.")
            else:
                gap = calculations['required_return'] - calculations['portfolio_expected_return']
                st.warning(f"⚠️ Your portfolio **DOES NOT** meet your minimum return requirements.")
                st.info(f"You need an additional **{gap*100:.2f}%** per year in portfolio return.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Portfolio Expected Return", f"{calculations['portfolio_expected_return']*100:.2f}%")
            with col2:
                st.metric("Required Return", f"{calculations['required_return']*100:.2f}%")
            
            st.markdown("---")
            
            # Options
            st.subheader("Options for Meeting Your Needs")
            
            tab1, tab2, tab3 = st.tabs(["💰 Change Savings", "📅 Adjust Timeline", "📊 Reallocate"])
            
            with tab1:
                st.markdown(f"""
                **OPTION 1: Change Your Savings Pattern**
                
                You could adjust your contributions to **${calculations['new_contribution']:,.2f}** per year to help fund your goal.
                
                - Current annual contribution: ${contributions:,.2f}
                - New annual contribution: ${calculations['new_contribution']:,.2f}
                - Difference: ${calculations['new_contribution'] - contributions:,.2f}
                """)
            
            with tab2:
                st.markdown(f"""
                **OPTION 2: Adjust Goal Date**
                
                If you adjust your goal by **{calculations['years_adjustment']:.1f} years**, you can still attain your funding requirement.
                
                - Current time horizon: {years} years
                - Adjusted time horizon: {calculations['adjusted_years']:.1f} years
                - New required return: {calculations['delayed_required_return']*100:.2f}%
                """)
            
            with tab3:
                st.markdown(f"""
                **OPTION 3: Reallocate Portfolio**
                
                You could consider reallocating your portfolio to maximize the probability of achieving your required return.
                
                **Suggested Allocation:**
                - Stocks: {calculations['suggested_stocks']*100:.2f}%
                - Bonds: {calculations['suggested_bonds']*100:.2f}%
                - Cash: {calculations['suggested_cash']*100:.2f}%
                
                **Current Allocation:**
                - Stocks: {stocks*100:.2f}%
                - Bonds: {bonds*100:.2f}%
                - Cash: {cash*100:.2f}%
                """)
            
            st.markdown("---")
            
            # Generate PDF
            st.subheader("📄 Download PDF Report")
            pdf_buffer = generate_pdf(client_name, inputs, calculations)
            
            st.download_button(
                label="Download Complete PDF Report",
                data=pdf_buffer,
                file_name=f"Risk_Analysis_{client_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

else:
    st.info("👈 Configure client information and portfolio details in the sidebar, then click **Generate Report**")
    
    # Show preview
    st.subheader("Report Preview")
    st.markdown("""
    This tool generates a comprehensive **HOW MUCH IS TOO MUCH** risk analysis report including:
    
    1. **Core Metrics**: Required Return, Portfolio Loss Tolerance, Recovery Return
    2. **Effects of Losses Chart**: Visual analysis of recovery requirements
    3. **Portfolio Analysis**: Current allocation and maximum losses
    4. **Return Gap Analysis**: Does your portfolio meet your needs?
    5. **Three Options**: Change savings, adjust timeline, or reallocate
    6. **Adjusted Allocation Scenario**: Impact of reallocation
    7. **Delayed Goal Scenario**: Impact of extending timeline
    8. **Full Disclosures**: Professional disclaimers and risk considerations
    
    **Instructions:**
    - Enter client name and portfolio details in the sidebar
    - Adjust current allocation percentages (must sum to 100%)
    - Click "Generate Report" to create the PDF
    """)

# Footer
st.markdown("---")
st.caption("HOW MUCH IS TOO MUCH Analysis Tool | Generated with Streamlit")
