from langgraph.graph import StateGraph
from typing import TypedDict, List
class FraudDetectionState(TypedDict):
    employee_id: str
    transactions: List[dict]
    anomalies: List[dict]
    risk_score: float
    investigation_notes: str
    evidence: List[dict]
    status: str
    

def collect_employee_transactions(employee_id: str) -> List[dict]:
    """
    Simulate collecting employee transactions from a database or API.
    In a real application, this would query a financial database.
    """
    return [
        {"date": "2023-10-01", "amount": 150.00, "description": "Office Supplies"},
        {"date": "2023-10-02", "amount": 200.00, "description": "Client Lunch"},
        {"date": "2023-10-03", "amount": 5000.00, "description": "Consulting Fee"},
        # Add more transactions as needed
    ]

def detect_spending_anomalies(transactions: List[dict]) -> List[dict]:
    """
    Simulate anomaly detection on transaction data.
    In a real application, this would use statistical methods or ML models.
    """
    anomalies = []
    for transaction in transactions:
        if transaction["amount"] > 1000:  # Example threshold
            anomalies.append(transaction)
    return anomalies

def check_policy_compliance(transactions: List[dict], anomalies: List[dict]) -> List[dict]:
    """
    Simulate checking policy compliance using LLM reasoning.
    In a real application, this would involve more complex logic.
    """
    policy_violations = []
    for anomaly in anomalies:
        if anomaly["amount"] > 1000:  # Example policy violation
            policy_violations.append(anomaly)
    return policy_violations

def calculate_fraud_risk(anomalies: List[dict], policy_violations: List[dict]) -> float:
    """
    Calculate a fraud risk score based on anomalies and policy violations.
    In a real application, this would involve more complex scoring logic.
    """
    risk_score = 0.0
    if anomalies:
        risk_score += len(anomalies) * 0.1  # Example scoring logic
    if policy_violations:
        risk_score += len(policy_violations) * 0.2  # Example scoring logic
    return min(risk_score, 1.0)  # Cap at 1.0
def generate_investigation_report(state: FraudDetectionState) -> str:
    """
    Generate an investigation report based on the state.
    In a real application, this would compile evidence and findings.
    """
    report = f"Investigation Report for Employee ID: {state['employee_id']}\n"
    report += f"Risk Score: {state['risk_score']}\n"
    report += "Anomalies Detected:\n"
    for anomaly in state["anomalies"]:
        report += f"- {anomaly['date']}: {anomaly['description']} (${anomaly['amount']})\n"
    report += "Policy Violations:\n"
    for violation in state["policy_violations"]:
        report += f"- {violation['date']}: {violation['description']} (${violation['amount']})\n"
    return report
# nodes
def data_ingestion_node(state: FraudDetectionState):
    # Collect and standardize transaction data
    transactions = collect_employee_transactions(state["employee_id"])
    return {"transactions": transactions}

def pattern_analysis_node(state: FraudDetectionState):
    # Run statistical analysis and anomaly detection
    anomalies = detect_spending_anomalies(state["transactions"])
    return {"anomalies": anomalies}

def policy_check_node(state: FraudDetectionState):
    # Verify policy compliance using LLM reasoning
    policy_violations = check_policy_compliance(
        state["transactions"], 
        state["anomalies"]
    )
    return {"policy_violations": policy_violations}

def risk_scoring_node(state: FraudDetectionState):
    # Calculate overall fraud risk score
    risk_score = calculate_fraud_risk(
        state["anomalies"],
        state["policy_violations"]
    )
    return {"risk_score": risk_score}

def investigation_node(state: FraudDetectionState):
    # Generate investigation report if high risk
    if state["risk_score"] > 0.7:
        report = generate_investigation_report(state)
        return {"investigation_notes": report, "status": "flagged"}
    return {"status": "cleared"}


# build edges

# def validate_travel_consistency(expense, travel_booking, calendar):
#     """
#     Cross-reference expense location with:
#     - Flight/hotel bookings
#     - Calendar appointments
#     - GPS/timestamp data
#     """
#     prompt = f"""
#     Analyze this travel expense for location consistency:
    
#     Expense: {expense['description']} at {expense['location']} on {expense['date']}
#     Flight Records: {travel_booking}
#     Calendar: {calendar}
    
#     Are there any red flags or inconsistencies?
#     """
#     return llm_chain.run(prompt)

# def analyze_meal_expenses(meals, employee_schedule):
#     """
#     Detect suspicious meal patterns:
#     - Multiple dinners same day
#     - Expensive meals during personal time
#     - Group meals with impossible attendee counts
#     """
#     for meal in meals:
#         if meal['amount'] > policy_limits['meal_max']:
#             # Use LLM to analyze if justification is reasonable
#             justification_analysis = analyze_expense_justification(meal)
            
#         # Check for duplicate restaurants/vendors
#         if detect_vendor_frequency_anomaly(meal['vendor']):
#             flag_potential_kickback_scheme(meal)

# def verify_team_event_attendees(event_expense, employee_data):
#     """
#     Cross-reference claimed attendees with:
#     - Badge access logs
#     - Calendar availability
#     - Team structure
#     """
#     claimed_attendees = extract_attendee_count(event_expense)
#     actual_team_size = get_team_size(event_expense['submitter'])
    
#     if claimed_attendees > actual_team_size * 1.5:  # 50% buffer for guests
#         return investigate_phantom_attendees(event_expense)

# def analyze_vendor_relationships(credit_card_transactions):
#     """
#     Detect potential kickback schemes:
#     - Repeated use of specific vendors
#     - Vendors outside normal business areas
#     - Unusual payment patterns
#     """
#     vendor_frequency = Counter(t['vendor'] for t in transactions)
    
#     for vendor, frequency in vendor_frequency.items():
#         if frequency > threshold:
#             # Use LLM to assess if vendor relationship seems legitimate
#             assessment = llm_chain.run(f"""
#             Analyze this vendor relationship for potential fraud:
#             Vendor: {vendor}
#             Frequency: {frequency} transactions
#             Employee: {employee_id}
#             Business Justification: {get_business_context(vendor)}
            
#             Rate suspicion level (1-10) and explain reasoning.
#             """)

# def benford_analysis(expense_amounts):
#     """
#     Apply Benford's Law to detect fabricated expenses
#     Natural expenses follow Benford's distribution
#     Fabricated amounts often don't
#     """
#     first_digits = [int(str(amount)[0]) for amount in expense_amounts]
#     benford_distribution = calculate_benford_expected()
#     actual_distribution = Counter(first_digits)
    
#     chi_square = calculate_chi_square_test(actual_distribution, benford_distribution)
    
#     if chi_square > threshold:
#         return "Suspicious: Expenses don't follow natural distribution"


# def analyze_submission_patterns(expense_submissions):
#     """
#     Detect suspicious timing patterns:
#     - End-of-month/quarter rushes
#     - Consistent round numbers
#     - Submissions just under approval limits
#     """
#     timing_analysis = {
#         'end_of_period_clustering': check_period_end_clustering(submissions),
#         'round_number_frequency': check_round_numbers(submissions),
#         'limit_gaming': check_approval_limit_gaming(submissions)
#     }
    
#     return generate_temporal_fraud_assessment(timing_analysis)


# Build the state graph
# Create the workflow graph
workflow = StateGraph(FraudDetectionState)

# Add nodes
workflow.add_node("data_ingestion", data_ingestion_node)
workflow.add_node("pattern_analysis", pattern_analysis_node)
workflow.add_node("policy_check", policy_check_node)
workflow.add_node("risk_scoring", risk_scoring_node)
workflow.add_node("investigation", investigation_node)

# Define the flow
workflow.add_edge("data_ingestion", "pattern_analysis")
workflow.add_edge("pattern_analysis", "policy_check")
workflow.add_edge("policy_check", "risk_scoring")
workflow.add_edge("risk_scoring", "investigation")

# Set entry point
workflow.set_entry_point("data_ingestion")

# Compile the workflow
app = workflow.compile()

from IPython.display import Image, display

try:
    mermaid_png_data  = app.get_graph().draw_mermaid_png()
    output_file_path = "my_graph_image.png"
    if not isinstance(mermaid_png_data, bytes):
        raise TypeError(
            "The draw_mermaid_png() method did not return bytes. "
            "Please ensure it returns raw PNG data."
        )

    # 2. Save the PNG data to a file
    with open(output_file_path, "wb") as image_file:
        image_file.write(mermaid_png_data)

    print(f"Graph image successfully saved to: {output_file_path}")
except Exception:
    # This requires some extra dependencies and is optional
    pass

# def stream_graph_updates(user_input: str):
#     for event in app.stream({"messages": [{"role": "user", "content": user_input}]}):
#         for value in event.values():
#             print("Assistant:", value["messages"][-1].content)

def stream_graph_updates(user_input: str):
    for event in app.stream({"employee_id": "12345", "messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break