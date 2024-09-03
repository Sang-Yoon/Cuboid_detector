import os
import rosbag2_py
from builtin_interfaces.msg import Time as BuiltinTime
import rclpy


def to_builtin_time(nanoseconds):
    """Convert nanoseconds to builtin_interfaces.msg.Time"""
    sec = int(nanoseconds // 1e9)
    nanosec = int(nanoseconds % 1e9)
    return BuiltinTime(sec=sec, nanosec=nanosec)


def to_seconds(builtin_time):
    """Convert builtin_interfaces.msg.Time to seconds"""
    return builtin_time.sec + builtin_time.nanosec / 1e9


def trim_bag_file(input_bag_path, output_bag_path, start_time_offset, end_time_offset):
    rclpy.init()

    bag_reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=input_bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    bag_reader.open(storage_options, converter_options)

    # Initialize variables
    first_time_ros2 = None
    start_time_sec = None
    end_time_sec = None

    # Read through all messages once to find the first timestamp
    if bag_reader.has_next():
        topic, msg, first_timestamp = bag_reader.read_next()
        first_time_ros2 = to_builtin_time(first_timestamp)
        first_time_sec = to_seconds(first_time_ros2)

        # Calculate the start and end times
        start_time_sec = first_time_sec + start_time_offset
        end_time_sec = first_time_sec + end_time_offset
    else:
        print("No messages found in the input bag file.")
        rclpy.shutdown()
        return

    print(f"Trimming messages between {start_time_sec} and {end_time_sec} seconds.")

    # Re-open the reader to start from the beginning
    bag_reader.open(storage_options, converter_options)

    # Create Bag Writer and open the output bag file
    bag_writer = rosbag2_py.SequentialWriter()
    output_storage_options = rosbag2_py.StorageOptions(uri=output_bag_path, storage_id="sqlite3")
    bag_writer.open(output_storage_options, converter_options)

    created_topics = {}

    while bag_reader.has_next():
        topic, msg, timestamp = bag_reader.read_next()

        timestamp_ros2 = to_builtin_time(timestamp)
        timestamp_sec = to_seconds(timestamp_ros2)

        # Adjust the topic creation logic
        if topic not in created_topics:
            topic_metadata = bag_reader.get_all_topics_and_types()
            for metadata in topic_metadata:
                if metadata.name == topic:
                    topic_info = rosbag2_py.TopicMetadata(
                        name=metadata.name,
                        type=metadata.type,
                        serialization_format=metadata.serialization_format,
                    )
                    bag_writer.create_topic(topic_info)
                    created_topics[topic] = topic_info
                    break

        # Ensure certain topics are always included (like extrinsics)
        if 'extrinsics' in topic or start_time_sec <= timestamp_sec <= end_time_sec:
            bag_writer.write(topic, msg, timestamp)

    rclpy.shutdown()


if __name__ == "__main__":
    input_bag_file = "/home/sangyoon/ros2_ws/src/Cuboid_detector/cuboid_detector/dataset/blue_box/L515_240723_0035/L515_240723_0035_0.db3"  # Replace with your input bag file path
    output_bag_file = "/home/sangyoon/ros2_ws/src/Cuboid_detector/cuboid_detector/dataset/blue_box/L515_240723_0035/output_bag_file.db3"  # Replace with your output bag file path
    if os.path.exists(output_bag_file):
        os.system(f"rm -rf {output_bag_file}")

    start_time_seconds = 15  # Replace with your desired start time in seconds
    end_time_seconds = 45  # Replace with your desired end time in seconds

    trim_bag_file(input_bag_file, output_bag_file, start_time_seconds, end_time_seconds)
