import os
import base64
import io
import threading
import queue
from matplotlib import use
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, jsonify
from werkzeug.utils import secure_filename
from scapy.all import *
import networkx as nx
import torch

# Import your specific modules
from pre_model import (
    processing_packet_conversion,
    predictingRowsCategory, 
    tokenizer, 
    model, 
    device, 
    classes
)

use('agg')

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pcap', 'pcapng'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for live capture
live_capture_running = False
packet_prediction_queue = queue.Queue()
capture_thread = None

# Error metrics and packets brief tracking
error_metrics = {'total_packets': 0}
packets_brief = {}
protocol_counts = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predictLivePackets(pkt):
    global error_metrics, packets_brief, protocol_counts
    
    if IP in pkt:
        # Protocol tracking
        if TCP in pkt:
            proto_name = 'TCP'
        else:
            proto_name = 'Other'
        protocol_counts[proto_name] = protocol_counts.get(proto_name, 0) + 1
        
        print(pkt[IP].src)
        if pkt[IP].src == '192.168.100.81':
            return
        try:
            input_line = processing_packet_conversion(pkt)
            if input_line is not None:
                truncated_line = input_line[:1024]
                tokens = tokenizer(truncated_line, return_tensors="pt")
                
                # Move input tensors to the appropriate device
                tokens = {k: v.to(device) for k, v in tokens.items()}
                model.to(device)
                
                # Perform prediction
                with torch.no_grad():
                    outputs = model(**tokens)
                    logits = outputs.logits
                    probabilities = logits.softmax(dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    predictedAttack = classes[predicted_class]
                
                # Update error metrics
                error_metrics['total_packets'] += 1
                
                # Track non-normal packets
                if predictedAttack != "Normal":
                    packets_brief[predictedAttack] = packets_brief.get(predictedAttack, 0) + 1
                    
                    print(f"Src IP: {pkt[IP].src}")
                    print(f"Predicted Attack: {predictedAttack}")
                
                # Put prediction data into queue for graph update
                packet_prediction_queue.put({
                    'protocol_counts': protocol_counts.copy(),
                    'packets_brief': packets_brief.copy()
                })
        
        except Exception as e:
            print(f"Prediction error: {e}")

def live_capture_worker(interface):
    global live_capture_running
    
    try:
        # Continuous packet sniffing
        sniff(iface=interface, prn=predictLivePackets, store=0, stop_filter=lambda x: not live_capture_running)
    except Exception as e:
        print(f"Live capture error: {e}")
    finally:
        live_capture_running = False

def generate_live_graphs(protocol_counts, packets_brief):
    """
    Generate base64 encoded graphs for live capture results
    """
    plt.figure(figsize=(15, 5))
    
    # Protocol Distribution Graph
    plt.subplot(1, 2, 1)
    plt.bar(protocol_counts.keys(), protocol_counts.values(), color='#341f97')
    plt.title('Live Protocol Distribution')
    plt.xlabel('Protocols')
    plt.ylabel('Packet Count')
    plt.xticks(rotation=45)
    
    # Potential Attacks Graph
    plt.subplot(1, 2, 2)
    plt.bar(packets_brief.keys(), packets_brief.values(), color='#ef6666')
    plt.title('Detected Attacks')
    plt.xlabel('Attack Types')
    plt.ylabel('Occurrence')
    plt.xticks(rotation=45)
    
    # Save graph to base64
    img_bytes = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    encoded_graph = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    plt.close()
    
    return f'<img src="data:image/png;base64,{encoded_graph}" alt="Live Network Analysis">'

def generate_graph(data, title, graph_color, xtext, ytext):
    plt.figure(figsize=(10, 5))
    plt.bar(data.keys(), data.values(), color=graph_color, width=0.7)
    plt.xlabel(xtext, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(ytext, weight='bold')
    plt.title(title)

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    # Convert the image to base64 encoding
    encoded_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    plt.close()

    return encoded_image

def create_network_graph(pcap_file):
    packets = rdpcap(pcap_file)
    G = nx.DiGraph()
    for packet in packets:
        try:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            G.add_edge(src_ip, dst_ip)
        except Exception as e:
            print(e)
    return G

def visualize_network_graph(pcap_file_path):

    network_graph = create_network_graph(pcap_file_path)

    pos = nx.spring_layout(network_graph)
    nx.draw(network_graph, pos, with_labels=True, font_size=8, node_size=1000, node_color='skyblue', font_color='black', font_weight='bold')
    #plt.show()

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    encoded_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    plt.close()

    return encoded_image

def visualize_destination_ports_plot(pcap_file_path, top_n=20):

    packets = rdpcap(pcap_file_path)

    destination_ports = {}

    for packet in packets:
        if IP in packet and TCP in packet:
            dst_ip = packet[IP].dst
            dst_port = packet[TCP].dport
            destination_ports[(dst_ip, dst_port)] = destination_ports.get((dst_ip, dst_port), 0) + 1
    sorted_ports = sorted(destination_ports.items(), key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(10, 6))

    top_ports = sorted_ports[:top_n]

    destinations, counts = zip(*top_ports)
    dst_labels = [f"{ip}:{port}" for (ip, port) in destinations]

    plt.bar(dst_labels, counts, color='skyblue')
    plt.xlabel('Destination IP and TCP Ports')
    plt.ylabel('Count')
    plt.title(f'Top {top_n} Most Contacted TCP Ports')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    encoded_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    plt.close()

    return encoded_image

# LIGHTGBM
import joblib
import numpy as np
import pandas as pd

# loading pickle objects
robust_scaler = joblib.load(r'LGBM\robust_scaler.pkl')
robust_scaler_features = ['Bytes', 'Bytes Sent', 'Bytes Received', 'Packets', 'Elapsed Time (sec)', 'pkts_sent', 'pkts_received', 'Host Source PR', 'Host Destination PR', 'NAT Source PR', 'NAT Destination PR']

std_scaler = joblib.load(r'LGBM\std_scaler.pkl')
std_scaler_features = ['Host CP', 'NAT CP', 'Host JI', 'NAT JI', 'Host SL', 'NAT SL', 'Host SI', 'NAT SI', 'Host AA', 'NAT AA']

calibrator_lgbm = joblib.load(r'LGBM\calibrator_lgbm.pkl')
HOST_NW = joblib.load(r'LGBM\host_nw.pkl')
NAT_NW = joblib.load(r'LGBM\nat_nw.pkl')


###################################################

# networkx library provides a function to calculate page rank of ports in the networks. It returns a dictionary where keys are ports and values are pageranks.
host_page_rank = nx.pagerank(HOST_NW)
nat_page_rank = nx.pagerank(NAT_NW)

def common_ports(nw, src, dst):
    """
    Counts no. of common ports connected directly between src and dst ports.

    Args:
        nw: Network instance
        src: Source Port
        dst: Destination Port

    Returns:
        No. of common ports from intersection of set of neighbors of both src and dst ports
    """
    try:
        return len(set(nw.neighbors(src)).intersection(set(nw.neighbors(dst))))
    except:
        return 0


def jaccard_index(nw, src, dst):
    """
    Counts Jaccard index between src and dst ports.

    Args:
        nw: Network instance
        src: Source Port
        dst: Destination Port

    Returns:
        Jaccard Index
    """
    try:
        return len(set(nw.neighbors(src)).intersection(set(nw.neighbors(dst)))) / len(set(nw.neighbors(src)).union(set(nw.neighbors(dst))))
    except:
        return 0


def salton_index(nw, src, dst):
    """
    Counts Salton index between src and dst ports.

    Args:
        nw: Network instance
        src: Source Port
        dst: Destination Port

    Returns:
        Salton Index
    """
    try:
        return len(set(nw.neighbors(src)).intersection(set(nw.neighbors(dst)))) / np.sqrt(len(set(nw.neighbors(src))) * len(set(nw.neighbors(dst))))
    except:
        return 0
        
def sorensen_index(nw, src, dst):
    """
    Counts Sorensen index between src and dst ports.

    Args:
        nw: Network instance
        src: Source Port
        dst: Destination Port

    Returns:
        Sorensen Index
    """
    try:
        return len(set(nw.neighbors(src)).intersection(set(nw.neighbors(dst)))) / (len(set(nw.neighbors(src))) + len(set(nw.neighbors(dst))))
    except:
        return 0

def adamic_adar_index(nw, src, dst):
    """
    Counts Adamic-Adar index between src and dst ports.

    Args:
        nw: Network instance
        src: Source Port
        dst: Destination Port

    Returns:
        Adamic-Adar Index
    """
    try:
        ports = set(nw.neighbors(src)).intersection(set(nw.neighbors(dst)))
        return 1/np.sum([np.log10(set(nw.neighbors(port))) for port in ports])
    except:
        return 0


def preprocess(feature_vector):
    """
    Preprocesses the feature matrix of firewall logs.

    Args:
        feature_vector: Input feature matrix of firewall logs

    Returns:
        Preprocessed feature vector
    """
    try:
        # reshaping into row vector if feature matrix of single instance is given
        # creating empty dataframe
        feature_names = ['Source Port', 'Destination Port', 'NAT Source Port', 'NAT Destination Port', 'Bytes', 'Bytes Sent',
                         'Bytes Received', 'Packets', 'Elapsed Time (sec)', 'pkts_sent', 'pkts_received']
        data_matrix = pd.DataFrame(feature_vector, columns = feature_names)
        # applying engineered features
        data_matrix['Source Port Translation'] = (data_matrix['Source Port'] != data_matrix['NAT Source Port']).astype('int')
        data_matrix['Destination Port Translation'] = (data_matrix['Destination Port'] != data_matrix['NAT Destination Port']).astype('int')
        data_matrix['Host CP'] = data_matrix.apply(lambda row: common_ports(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
        data_matrix['NAT CP'] = data_matrix.apply(lambda row: common_ports(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)
        data_matrix['Host JI'] = data_matrix.apply(lambda row: jaccard_index(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
        data_matrix['NAT JI'] = data_matrix.apply(lambda row: jaccard_index(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)
        data_matrix['Host SL'] = data_matrix.apply(lambda row: salton_index(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
        data_matrix['NAT SL'] = data_matrix.apply(lambda row: salton_index(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)
        data_matrix['Host SI'] = data_matrix.apply(lambda row: sorensen_index(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
        data_matrix['NAT SI'] = data_matrix.apply(lambda row: sorensen_index(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)
        data_matrix['Host AA'] = data_matrix.apply(lambda row: adamic_adar_index(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
        data_matrix['NAT AA'] = data_matrix.apply(lambda row: adamic_adar_index(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)
        data_matrix['Host Source PR'] = data_matrix.apply(lambda row: host_page_rank.get(row['Source Port'], 0), axis = 1)
        data_matrix['Host Destination PR'] = data_matrix.apply(lambda row: host_page_rank.get(row['Destination Port'], 0), axis = 1)
        data_matrix['NAT Source PR'] = data_matrix.apply(lambda row: nat_page_rank.get(row['NAT Source Port'], 0), axis = 1)
        data_matrix['NAT Destination PR'] = data_matrix.apply(lambda row: nat_page_rank.get(row['NAT Destination Port'], 0), axis = 1)
        # scaling the data
        data_matrix_robust_scaled = robust_scaler.transform(data_matrix[robust_scaler_features])
        data_matrix_std_scaled = std_scaler.transform(data_matrix[std_scaler_features])
        data_matrix_preprocessed = np.hstack((data_matrix[data_matrix.columns[:4]], data_matrix_robust_scaled, data_matrix_std_scaled, data_matrix[['Source Port Translation', 'Destination Port Translation']]))
        return data_matrix_preprocessed
    except:
        print("The last dimension of the data should be 11")



@app.route("/")
def home():
    return render_template('index.html')

@app.route('/live_capture')
def live_capture():
    return render_template('live_capture.html')

@app.route("/BERT")
def BERT():
    return render_template('BERT.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global packets_brief, protocol_counts
    
    # Reset tracking variables
    packets_brief.clear()
    protocol_counts.clear()

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    # filter_value = request.form['filter']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        packets_brief, protocol_counts = predictingRowsCategory(file_path)

        # Generate graphs
        graph1 = generate_graph(packets_brief, 'Identified Known Attacks', '#ef6666', "Attacks", "Number of Malicious Packets")
        graph2 = generate_graph(protocol_counts, 'Identified Protocols', '#341f97', "Protocols", "Number of Packets")

        # Generate Third graph

        graph3 = visualize_network_graph(file_path)

        # Generate Fourth graph

        graph4 = visualize_destination_ports_plot(file_path)

        # Determine alert color and text
        if len(packets_brief) > 0:
            alert_color = "#c0392b"
            alert_text = f"{filename} contains malicious network activity!"
        else:
            alert_color = "#27ae60"
            alert_text = f"{filename} is clear! 👌"

        return render_template('response.html', 
                               graph1=graph1, 
                               graph2=graph2,
                               graph3=graph3,
                               graph4=graph4,
                               alert_color=alert_color, 
                               alert_text=alert_text)

@app.route('/start_live_capture', methods=['POST'])
def start_live_capture():
    global live_capture_running, capture_thread
    
    # Get network interface from request or use a default
    interface = request.form.get('interface', 'Wi-Fi')
    
    if not live_capture_running:
        # Reset global tracking variables
        global error_metrics, packets_brief, protocol_counts
        error_metrics = {'total_packets': 0}
        packets_brief = {}
        protocol_counts = {}
        
        # Start capture thread
        live_capture_running = True
        capture_thread = threading.Thread(target=live_capture_worker, args=(interface,))
        capture_thread.start()
        
        return jsonify({'status': 'started'})
    
    return jsonify({'status': 'already_running'})

@app.route('/get_live_data', methods=['GET'])
def get_live_data():
    global live_capture_running, packet_prediction_queue
    
    try:
        if not packet_prediction_queue.empty():
            # Get the most recent prediction data
            while not packet_prediction_queue.empty():
                data = packet_prediction_queue.get(block=False)
            
            graphs = generate_live_graphs(
                data['protocol_counts'], 
                data['packets_brief']
            )

            return jsonify({
                'status': 'running' if live_capture_running else 'stopped',
                'graphs': graphs
            })
        else:
            return jsonify({
                'status': 'running' if live_capture_running else 'stopped',
                'graphs': None
            })
    except Exception as e:
        print(f"Error retrieving live data: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_live_capture', methods=['POST'])
def stop_live_capture():
    global live_capture_running, capture_thread
    
    live_capture_running = False
    if capture_thread:
        capture_thread.join()
    
    # Clear the prediction queue
    while not packet_prediction_queue.empty():
        packet_prediction_queue.get()
    
    return jsonify({'status': 'stopped'})

@app.route('/LGBM', methods=['GET'])
def lgbm_home():
    return render_template("LGBM.html")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the firewall Action depending on the feature_matrix of firewall logs

    Returns:
        Predicted Action class
    """
    inp = request.form.to_dict()
    feature_vector = np.array([inp['source_p'], inp['dest_p'], inp['nat_source_p'], inp['nat_dest_p'], inp['bytes'], inp['bts_sent'], inp['bts_received'], inp['pkts'], inp['elapsed_t'], inp['pkts_sent'], inp['pkts_received']])
    data_preprocessed = preprocess(feature_vector.reshape(1, -1))
    prediction = calibrator_lgbm.predict(data_preprocessed.reshape(1, -1))
    return jsonify({'Predicted Action': str(prediction)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)