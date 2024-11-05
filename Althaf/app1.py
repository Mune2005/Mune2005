from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Sample list of rooms (for demonstration purposes, limited to 10 rooms)
rooms = {f"Room_{i}": None for i in range(1, 11)}  # Dictionary to store room assignments

# Function to assign room to a patient
def allot_room(patient_name):
    # Check if the patient already has a room
    for room, occupant in rooms.items():
        if occupant == patient_name:
            return f"{patient_name} is already allotted to {room}."
    
    # Find an available room
    for room, occupant in rooms.items():
        if occupant is None:
            rooms[room] = patient_name
            return f"{patient_name} has been allotted to {room}."
    
    # If no room is available
    return "No rooms available."

# Function to release a room
def release_room(patient_name):
    for room, occupant in rooms.items():
        if occupant == patient_name:
            rooms[room] = None
            return f"{patient_name} has been removed from {room}."
    return f"{patient_name} was not found in any room."

@app.route("/")
def index():
    return render_template("sample.html")

# API endpoint for room allotment
@app.route("/allot", methods=["POST"])
def allot():
    data = request.get_json()
    patient_name = data.get("patient_name")
    
    if not patient_name:
        return jsonify({"error": "Patient name is required."}), 400
    
    result = allot_room(patient_name)
    return jsonify({"result": result})

# API endpoint for releasing a room
@app.route("/release", methods=["POST"])
def release():
    data = request.get_json()
    patient_name = data.get("patient_name")
    
    if not patient_name:
        return jsonify({"error": "Patient name is required."}), 400
    
    result = release_room(patient_name)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
