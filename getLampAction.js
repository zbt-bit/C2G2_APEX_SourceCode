async function getLampAction(isNight, isMotion, weather) {
    const response = await fetch('http://127.0.0.1:5000/api/control_lamp', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            is_night_time: isNight,
            is_motion_detected: isMotion,
            weather_data: weather
        })
    });
    
    if (response.ok) {
        const result = await response.json();
        console.log("Lamp Action:", result.lamp_action);
        return result.lamp_action;
    } else {
        console.error("API Error:", response.statusText);
        return "ERROR";
    }
}

// Example usage on the frontend: Simulating a rainy night with no motion
const rainyNightWeather = {
    humidity: 95, 
    cloudcover: 80, 
    visibility: 5, 
    uvindex: 1, 
    day_of_year: 300, // Replace with current day of year
    temp: 25, 
    precip: 15.0
};

// This call would result in: MAX OUTPUT (Safety Override: Rain/Fog)
getLampAction(true, false, rainyNightWeather);