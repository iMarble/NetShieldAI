from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from scapy.all import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

classes = [
    "Analysis",
    "Backdoor",
    "Bot",
    "DDoS",
    "DoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS SlowHTTPTest",
    "DoS Slowloris",
    "Exploits",
    "FTP Patator",
    "Fuzzers",
    "Generic",
    "Heartbleed",
    "Infiltration",
    "Normal",
    "Port Scan",
    "Reconnaissance",
    "SSH Patator",
    "Shellcode",
    "Web Attack - Brute Force",
    "Web Attack - SQL Injection",
    "Web Attack - XSS",
    "Worms",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model and tokenizer
model_path = r"fine_tuned_model"  # Path to your saved model directory
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Initialize dictionaries and lists for packet analysis.
packets_brief = {}
forward_packets = {}
backward_packets = {}
protocols = []
protocol_counts = {}

# Error metrics dictionary
error_metrics = {
    'total_packets': 0,
    'misclassified_packets': 0,
    'error_percentages': {}
}

def visualize_attack_distribution(packets_brief):
    """
    Create a bar graph of detected attacks
    """
    plt.figure(figsize=(15, 6))
    keys = list(packets_brief.keys())
    vals = list(packets_brief.values())
    
    # Create bar plot with seaborn
    sns.barplot(x=keys, y=vals)
    plt.xlabel('Attack Types')
    plt.ylabel('Number of Packets')
    plt.title('Detected Network Attacks')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def calculate_error_percentages():
    """
    Calculate and visualize error percentages
    """
    total_packets = error_metrics['total_packets']
    misclassified = error_metrics['misclassified_packets']
    
    # Overall error rate
    overall_error_rate = (misclassified / total_packets) * 100 if total_packets > 0 else 0
    
    print(f"\nError Analysis:")
    print(f"Total Packets Analyzed: {total_packets}")
    print(f"Misclassified Packets: {misclassified}")
    print(f"Overall Error Rate: {overall_error_rate:.2f}%")
    
    # Visualize error percentages
    plt.figure(figsize=(10, 6))
    plt.pie([overall_error_rate, 100 - overall_error_rate], 
            labels=['Error Rate', 'Correct Classification'], 
            autopct='%1.1f%%')
    plt.title('Model Classification Accuracy')
    plt.show()

def trainFromPcapFile(file_path, label, application_filter=None):
    training_set = []
    train_labels = []

    with PcapReader(file_path) as pcap:
        for pkt in pcap:
            if IP in pkt and TCP in pkt:  # IPv4 and TCP
                payload_bytes_to_filter = bytes(pkt.payload)
                if (
                    application_filter is None
                    or application_filter in payload_bytes_to_filter
                ):
                    input_line = processing_packet_conversion(pkt)
                    if input_line is not None:
                        truncated_line = input_line[:1024]
                        training_set.append(truncated_line)
                        train_labels.append(label)

    tokenized_input = tokenizer(
        training_set, padding=True, truncation=True, return_tensors="pt"
    )
    tokenized_input["labels"] = torch.tensor(train_labels)

    model.to(device)
    # Move input tensors to the specified device
    tokenized_input = {key: value.to(device) for key, value in tokenized_input.items()}

    # Data loader
    dataset = TensorDataset(
        tokenized_input["input_ids"],
        tokenized_input["attention_mask"],
        tokenized_input["labels"],
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    num_training_samples = len(dataloader.dataset)
    print(f"Number of training samples: {num_training_samples}")

    # Optimizer and loss function
    optimizer = Adam(model.parameters(), lr=1e-5)
    criterion = CrossEntropyLoss()

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for iteration, batch in enumerate(dataloader, 1):
            input_ids, attention_mask, labels = batch

            # Move batch tensors to the specified device
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predictions = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            print(f"Total samples: {total_samples}")

        model.save_pretrained("fine_tuned_model")

def processing_packet_conversion(packet):
    # Clone the packet for processing without modifying the original.
    packet_2 = packet

    while packet_2:
        # Extract and count protocol layers in the packet.
        layer = packet_2[0]
        if layer.name not in protocol_counts:
            protocol_counts[layer.name] = 0
        else:
            protocol_counts[layer.name] += 1
        protocols.append(layer.name)

        # Break if there are no more payload layers.
        if not layer.payload:
            break
        packet_2 = layer.payload

    # Extract relevant information for feature creation.
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    src_port = packet.sport
    dst_port = packet.dport
    ip_length = len(packet[IP])
    ip_ttl = packet[IP].ttl
    ip_tos = packet[IP].tos
    tcp_data_offset = packet[TCP].dataofs
    tcp_flags = packet[TCP].flags

    # Process payload content and create a feature string.
    payload_bytes = bytes(packet.payload)
    payload_length = len(payload_bytes)
    payload_content = payload_bytes.decode("utf-8", "replace")
    payload_decimal = " ".join(str(byte) for byte in payload_bytes)
    final_data = (
        "0"
        + " "
        + "0"
        + " "
        + "195"
        + " "
        + "-1"
        + " "
        + str(src_port)
        + " "
        + str(dst_port)
        + " "
        + str(ip_length)
        + " "
        + str(payload_length)
        + " "
        + str(ip_ttl)
        + " "
        + str(ip_tos)
        + " "
        + str(tcp_data_offset)
        + " "
        + str(int(tcp_flags))
        + " "
        + "-1"
        + " "
        + str(payload_decimal)
    )
    return final_data

def predictingRowsCategory(file_path):
    packets_nbr = 0  # Initialize packet counter
    with PcapReader(file_path) as pcap:
        for pkt in pcap:
            if IP in pkt:  # Check for IPv4 packets
                if TCP in pkt:
                    input_line = processing_packet_conversion(
                        pkt
                    )  # Process packet data
                    if input_line is not None:
                        truncated_line = input_line[:1024]  # Limit input length
                        tokens = tokenizer(
                            truncated_line, return_tensors="pt"
                        )  # Tokenize input

                        # Move input tensors to the appropriate device (CPU or GPU)
                        tokens = {k: v.to(device) for k, v in tokens.items()}
                        model.to(device)  # Move the model to the same device

                        outputs = model(**tokens)  # Pass tokens through the model
                        logits = outputs.logits
                        probabilities = logits.softmax(
                            dim=1
                        )  # Calculate class probabilities
                        predicted_class = torch.argmax(
                            probabilities, dim=1
                        ).item()  # Get predicted class index

                        predictedAttack = classes[
                            predicted_class
                        ]  # Map index to corresponding attack class

                        # Update error metrics
                        error_metrics['total_packets'] += 1

                        if predictedAttack != "Normal":
                            # Update or add count for non-normal packets in packets_brief dictionary
                            if predictedAttack not in packets_brief:
                                packets_brief[predictedAttack] = 1
                            else:
                                packets_brief[predictedAttack] += 1

                        # Print prediction details
                        print("Predicted class:", predicted_class)
                        print("predicted class is : ", classes[predicted_class])
                        print("Class probabilities:", probabilities.tolist())

                    packets_nbr += 1  # Increment packet counter

    return packets_brief, protocol_counts
    # Visualize attack distribution
    # visualize_attack_distribution(packets_brief)
    
    # Calculate and show error percentages
    # calculate_error_percentages()

def predictLivePackets(pkt):
    if IP in pkt:  # Check for IPv4 packets
        if TCP in pkt:
            input_line = processing_packet_conversion(pkt)  # Process packet data
            if input_line is not None:
                truncated_line = input_line[:1024]  # Limit input length
                tokens = tokenizer(
                    truncated_line, return_tensors="pt"
                )  # Tokenize input
                
                # Move input tensors to the appropriate device (CPU or GPU)
                tokens = {k: v.to(device) for k, v in tokens.items()}
                model.to(device)  # Move the model to the same device
                
                outputs = model(**tokens)  # Pass tokens through the model
                logits = outputs.logits
                probabilities = logits.softmax(dim=1)  # Calculate class probabilities
                predicted_class = torch.argmax(
                    probabilities, dim=1
                ).item()  # Get predicted class index
                predictedAttack = classes[
                    predicted_class
                ]  # Map index to corresponding attack class
                
                # Update error metrics
                error_metrics['total_packets'] += 1

                if predictedAttack != "Normal":
                    # Update or add count for non-normal packets in packets_brief dictionary
                    if predictedAttack not in packets_brief:
                        packets_brief[predictedAttack] = 1
                    else:
                        packets_brief[predictedAttack] += 1
                
                # print("Src IP is:", pkt[IP].src)
                # print("Predicted class:", predicted_class)
                # Print prediction details
                if predictedAttack != "Normal":
                    print("Src IP is:", pkt[IP].src)
                    print("Predicted class:", predicted_class)
                    print("predicted class is : ", classes[predicted_class])

def start_firewall():
    print("Starting the firewall...")
    try:
        # Sniff packets and call packet_callback for each packet
        sniff(iface="Wi-Fi", prn=predictLivePackets, store=0)
    except KeyboardInterrupt:
        print("\nFirewall terminated by user.")
        
        # Visualize results after firewall is stopped
        visualize_attack_distribution(packets_brief)
        calculate_error_percentages()

if __name__ == "__main__":
    #start_firewall()
    predictingRowsCategory(r"D:\CS\AI\PcapFileAnalysis\PcapSamples\xmas.pcap")