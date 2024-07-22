from flask import jsonify, request, render_template
from app import app
from lib.ultrasonic_sensor import measure_distance, set_servo_angle
import logging

logging.basicConfig(filename='../logs/app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    data = request.get_json()
    angle = data.get('angle', 90)
    max_range = data.get('max_range', 80)

    logger.debug(f"Received scan request at angle: {angle}, range: {max_range}")
    set_servo_angle(angle)
    distance = measure_distance(max_range)

    if distance is not None:
        response = {'angle': angle, 'distance': distance}
        logger.info(f"Scan successful: {response}")
    else:
        response = {'angle': angle, 'distance': 'Error'}
        logger.error("Scan error: Distance measurement failed")

    return jsonify(response)
