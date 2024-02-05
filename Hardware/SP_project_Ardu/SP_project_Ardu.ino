#include "DHT.h"
#include <SoftwareSerial.h>
#define DHTPIN 2
#define Delay1 1
// Uncomment whatever type you're using!
#define DHTTYPE DHT11 // DHT 11
//#define DHTTYPE DHT22 // DHT 22 (AM2302), AM2321
//#define DHTTYPE DHT21 // DHT 21 (AM2301)
int sensorPin2 = A1;
int sensorPin = A0;
int sensorPin3 = A2;
SoftwareSerial espSerial(5, 6);
DHT dht(DHTPIN, DHTTYPE);
String str;
int stt;
void setup(){
Serial.begin(115200);
espSerial.begin(115200);
  pinMode(A0, INPUT);
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(1, OUTPUT);
delay(2000);
}
void loop()
{
 int sensorValue = analogRead(sensorPin);
 int sensorValue2 = analogRead(sensorPin2);
 int sensorValue3 = analogRead(sensorPin3);
Serial.print("sen1: ");
sensorValue = map(sensorValue, 0, 1024, 100, 0);
Serial.print(sensorValue);
Serial.print(" ");
Serial.print("sen2: ");
Serial.print(sensorValue2);
sensorValue2 = map(sensorValue2, 0, 1024, 100, 0);
Serial.print(" ");
Serial.print("sen3: ");
Serial.print(sensorValue3);
sensorValue3 = map(sensorValue3, 0, 1024, 100, 0);
Serial.println(" ");
//str =String("coming from arduino: ")+String("H= ")+String(sensorValue)+String("T= ")+String(sensorValue2);
//espSerial.println(str);
espSerial.write(sensorValue);
espSerial.write(sensorValue2);
espSerial.write(sensorValue3);
delay(30000);
//if(sensorValue < 40 &&  sensorValue2 < 40){
//  while( sensorValue < 80 &&  sensorValue2 < 80){
//  digitalWrite(Delay1, HIGH);
//  sensorValue = analogRead(sensorPin);
//  sensorValue2 = analogRead(sensorPin2);
//  }
//  digitalWrite(Delay1, LOW);
//}
}
