require('dotenv').config();
const mqtt = require('mqtt');
const { createClient } = require('@supabase/supabase-js');

// Initialize Supabase client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

// MQTT Configuration
const mqttOptions = {
  username: process.env.MQTT_USERNAME,
  password: process.env.MQTT_PASSWORD,
  port: 8883,
  rejectUnauthorized: false 
};

// Initialize MQTT client
const client = mqtt.connect(process.env.MQTT_BROKER_URL, mqttOptions);

// Store received values and their timestamps
let sensorData = {
  temperature: { value: null, timestamp: null },
  humidity: { value: null, timestamp: null },
  fan_status: { value: null, timestamp: null }
};

// Function to insert data into Supabase
async function insertSensorData() {
  try {
    const currentTime = new Date();
    
    // Find the most recent timestamp among all sensors
    const timestamps = Object.values(sensorData)
      .map(data => data.timestamp)
      .filter(timestamp => timestamp !== null);
    
    if (timestamps.length === 0) return;
    
    const mostRecentTimestamp = new Date(Math.max(...timestamps));
    
    // Only insert if we have data within the last 5 seconds
    if (currentTime - mostRecentTimestamp > 5000) return;
    
    const dataToInsert = {
      temperature: sensorData.temperature.value,
      humidity: sensorData.humidity.value,
      fan_status: sensorData.fan_status.value,
      recorded_at: mostRecentTimestamp.toISOString()
    };
    
    console.log('Inserting data:', dataToInsert);
    
    const { data, error } = await supabase
      .from('sensor_data')
      .insert([dataToInsert])
      .select();
    
    if (error) {
      console.error('Error inserting data:', error);
    } else {
      console.log('Successfully inserted data with ID:', data[0].id);
    }
  } catch (error) {
    console.error('Error in insertSensorData:', error);
  }
}

// MQTT Connection handling
client.on('connect', () => {
  console.log('Connected to MQTT broker');
  
  // Subscribe to topics
  client.subscribe('device/temperature', (err) => {
    if (err) console.error('Error subscribing to temperature:', err);
  });
  
  client.subscribe('device/humidity', (err) => {
    if (err) console.error('Error subscribing to humidity:', err);
  });
  
  client.subscribe('device/fan', (err) => {
    if (err) console.error('Error subscribing to fan:', err);
  });
});

client.on('error', (error) => {
  console.error('MQTT Error:', error);
});

// Handle incoming messages
client.on('message', async (topic, message) => {
  try {
    const value = message.toString();
    const currentTime = new Date();
    
    switch (topic) {
      case 'device/temperature':
        sensorData.temperature = { value: parseFloat(value), timestamp: currentTime };
        break;
      case 'device/humidity':
        sensorData.humidity = { value: parseFloat(value), timestamp: currentTime };
        break;
      case 'device/fan':
        sensorData.fan_status = { value: value, timestamp: currentTime };
        break;
    }
    
    // Try to insert data whenever we receive a new value
    await insertSensorData();
    
  } catch (error) {
    console.error('Error processing message:', error);
  }
});

// Handle process termination
process.on('SIGINT', () => {
  console.log('Disconnecting from MQTT broker...');
  client.end();
  process.exit();
}); 