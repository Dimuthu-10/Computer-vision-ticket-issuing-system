# Computer Vision ticket Issuing System

### Introduction
An automated ticket issuing system is a system that is powered by computer vision. In a traditional ticket issuing system, the user has to enter the vehicle number and other specific details manually in the system. Mainly most of the parking ticket issuing system has a vehicle number as the main input. The user has to enter it manually by asking the vehicle owner or look by himself. And also, the system user has to identify the vehicle type and update the bill with the vehicle type to generate the bill for the vehicle type. But through this project, I have proposed an automated ticket issuing system that the capturing vehicle number plate and detects the vehicle type with computer vision, and stores them in the database. And issuing a ticket related to the vehicle types and the related to the vehicle number. And also develop a mobile application for highway users to make them more familiar and keep zero contact with other people. Through that mobile application highways, users can make payments by scanning QR codes. In a pandemic situation like this, it is important to keep minimum contact with another person. Therefore, this automated ticket issuing system plays a major role to keep minimum contact between two people and reducing significant paper costs through this system. Because of this pandemic situation. All the activities are getting automated step by step. Therefore, this will be the most suitable project for the present pandemic situation.

### Scope of the completed project
When considering the whole project scope, it has a very large project scope complete and archives through the functionalities. But as the main functionalities can be identified few main functionalities. Because all the other functionalities depend on the main functionalities. When listing downing the main functionalities.
- The first one is identifying the vehicle type and detecting the vehicle type when the vehicle is entering the premises or the road. Because most of the ticket issuing systems the ticket prices are depending on the vehicle type. Therefore, the application should identify the vehicle type and decide what the vehicle is in what category and decide the price that needs to pay.
- The second one is identifying the vehicle license plate and reading the license plate numbers through the application. Because the system needs to identify each vehicle using a number plate. Therefore, the system needs to read the license plate data related to each and every entry of the vehicle.
- The third one is to retrieve the stored data related to the vehicle categories and stored data related to each and every vehicle number. Because as I mentioned in the first functionality application identifies the vehicle type and should be retrieved data from the database related to identified vehicle category. And also the system needs to keep records for entry of each and every vehicle for analysis purposes and also the prediction purpose.

These things are the main functionalities of the proposed system. As I mentioned earlier the other functionalities are additional and simple for this system. Therefore, I have decided to complete the main functionalities in the given time. Because when trying to archive all the functionalities takes a much longer time period to complete. I should have mentioned here that those functionalities are not achievable in the given time period. Because it takes time to archive. That is why I need to focus on the main functionalities in the given time period. When listing down the other simple functionalities,
- Create a payment gateway to achieve mobile payment using QR code.
- Bill printing functionality.
- Fixed location functionality.
- Introduce the mobile app for parking users to achieve mobile payments.
- Introduce the prepaid recharge card for the daily users who use parking or the highway.

### Specification and Design
The system should provide the following services as intended. These are the primary goals the system is expected to achieve.
- Detect the vehicle and identify the vehicle number plate.
- Detect the vehicle license plate and read the characters of that.
- Store the entrance details in the relevant database
- At the exit point identify again the vehicle and retrieve the entry details to calculate the price and create the bill.
- Using those details generate the bill.

These are the most basic expectations of the system.

#### SRS Diagram 
<!-- ![srs diagram](https://user-images.githubusercontent.com/58289018/170037033-b9aaadbe-731b-4b38-9c04-25d3f430cc51.png | width=400) -->
<img src="https://user-images.githubusercontent.com/58289018/170037033-b9aaadbe-731b-4b38-9c04-25d3f430cc51.png" width="600" height="400">

### Functional Requirements
- Capture the vehicle type
- Capture the license plate
- Read the license plate
- Store the vehicle type, license plate, and the entered time in the database
- Capture again license plate at the exit
- Read the license plate and retrieve the stored data using a query
- Calculate the bill for each vehicle
- Print the bill

### Non-Functional Requirements
- Performance should be fast and correct. That is why it is important to use a virtual environment to deploy the application.
- Availability is an important part of this system. Because this system is a real-time application. We can’t shut it down after the deployment. Therefore, we have to use a backup power supply and stable network connection to keep availability.
- Reliability is also the main part of this system. Because the number plate reading part is the main input of this system.
- Capacity is also the main part. A considerable number of vehicles is entering the highway in one hour.

### Future work
When talking about the future implantation, there are lots of things to do to perform a fully automated system. Because this system must be reliable and faster than the existing systems. Therefore, future works need to do the things below
- Deploy the developed system into the virtual environment.
- Create and web-based creative user interface.
- Develop the existing architecture with high accuracy.
- Develop a mobile application for vehicle owners.
- Create a payment gateway to perform transactions.
- Use a real-time detection method to identify the vehicles.
- Train existing CNN with the heavy and clean database related to our country vehicles.
