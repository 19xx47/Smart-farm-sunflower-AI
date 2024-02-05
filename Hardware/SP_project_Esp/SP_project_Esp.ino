#include <ESP8266WiFi.h>
#include "ThingSpeak.h"

const char* ssid = "Mai_2.4GHz";   // your network SSID (name)
const char* password = "Nok07200";   // your network password
//const char* ssid = "12345678";   // your network SSID (name)
//const char* password = "1212312121";   // your network password
WiFiClient client;
unsigned long myChannelNumber = 1;
const char * myWriteAPIKey = "R3TI7UIJ8HNO4LBH";
unsigned long lastTime = 0;
unsigned long timerDelay = 30000;
void setup() {
// Open serial communications and wait for port to open:
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);


  ThingSpeak.begin(client);  // Initialize ThingSpeak
while (!Serial) {
; // wait for serial port to connect. Needed for native USB port only
}
}
void loop() { // run over and over
  int saturation = 0;
  int rate_heart = 0;
  int sensor3 = 0;
  if ((millis() - lastTime) > timerDelay) {

    // Connect or reconnect to WiFi
    if(WiFi.status() != WL_CONNECTED){
      Serial.print("Attempting to connect");
      while(WiFi.status() != WL_CONNECTED){
        WiFi.begin(ssid, password);
        delay(5000);
      }
      Serial.println("\nConnected.");
    }

  if (Serial.available()) {
    saturation = Serial.read();
  }
  Serial.print("sensor1 =  ");
  Serial.print(saturation);
  Serial.print("    ");

  if (Serial.available()) {
    rate_heart = Serial.read();

  }
  Serial.print("sensor2 =  ");
  Serial.print(rate_heart);
  Serial.print("    ");

  if (Serial.available()) {
   sensor3 = Serial.read();
  }
  Serial.print("sensor3 =  ");
  Serial.print(sensor3);


  ThingSpeak.setField(1, saturation);
  ThingSpeak.setField(2, rate_heart);
  ThingSpeak.setField(3, sensor3);
  int x = ThingSpeak.writeFields(myChannelNumber, myWriteAPIKey);

    if(x == 200){
      Serial.println("Channel update successful.");
    }
    else{
      Serial.println("Problem updating channel. HTTP error code " + String(x));
    }
    lastTime = millis();
  }
}
