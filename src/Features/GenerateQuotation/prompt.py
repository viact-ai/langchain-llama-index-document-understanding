PRICING_TABLE_PROMPT = """product  product_id  descriptions    price   pricing_unit    category
Admin user account login on viAct.ai	G001	Admin user account login including set up w/ 5 standard user account	 $100.00 	/month	General Items
Camera set up and installation	G003	include only accessible location, power and internet connection	 $3,000.00 	/set	General Items
Customized AI engine	D011	Usage of customized AI detection module (after Item 10.0 activated)	 $1,000.00 	/month	Detection Modules
Customized AI engine development	D012	As per requirement	 $10,000.00 	or up	Detection Modules
Customized AI engine training	D013	Under 3,000 images	 $1,000.00 	/ training	Detection Modules
Customized dashboard, report and AI on viAct.ai with unique web link	G002	As per requirement	 $20,000.00 	or up	General Items
Danger Zone Alert Sensoring System (1 sensor)	P01	1 sensor head, 1 processing unit, 1 screen, 15 meters cables between sensor head and processing unit.	 $1,000.00 	/month	Hardware Modules
Danger Zone Alert Sensoring System (2 sensors)	P02	2 sensor heads, 1 processing unit, 1 screen, 15 meters cables between sensor head and processing unit.	 $1,100.00 	/month	Hardware Modules
Danger Zone Alert Sensoring System (3 sensors)	P03	3 sensor heads, 1 high performance processing unit, 1 screen, 15 meters cables between sensor head and processing unit.	 $1,200.00 	/month	Hardware Modules
Danger Zone Alert Sensoring System (4 sensors)	P04	4 sensor heads, 1 high performance processing unit, 1 screen, 15 meters cables between sensor head and processing unit.	 $1,300.00 	/month	Hardware Modules
General AI engine	D001	Detection on human with or without safety helmet	 $200.00 	/month	Detection Modules
General AI engine	D002	Detection on human with or wihtout safety jacket	 $200.00 	/month	Detection Modules
General AI engine	D003	Detection on the color of safety wear on worker to classify company they work for	 $200.00 	/month	Detection Modules
General AI engine	D004	Detection on human entering danger zone	 $200.00 	/month	Detection Modules
General AI engine	D005	Detection on machinery (crane, forklift, tower crane) and duration	 $200.00 	/month	Detection Modules
General AI engine	D006	Detection on human with or without smoking	 $300.00 	/month	Detection Modules
General AI engine	D007	Detection on vehicle/ dump truck in & out	 $300.00 	/month	Detection Modules
General AI engine	D008	Detection on human in & out and duration	 $300.00 	/month	Detection Modules
General AI engine	D009	Detection on human dumping tash on spot	 $300.00 	/month	Detection Modules
General AI engine	D010	Detection on trash classification (1 category)	 $300.00 	/month	Detection Modules
Multi cameras and NVR set up and installation	G004	As per requirement	 $3,0000.00 	or up	General Items
On site technical support	G006	1 day (8 hours) on site technical support with specialized engineer	 $500.00 	/day	General Items
On site technical training	G005	1 day (8 hours) on site technical training with specialized trainer	 $500.00 	/day	General Items
Other support	G007	Other set up and maintenance support	 Per requirement 	 Per requirement 	General Items
Rental of AI processor	P05	As per requirement	 $250.00 	/month	Hardware Modules
"""

############

SYSTEM_PROMPT = """You are an helpful Assistant that able to generated quoatation based on user requirements.
User will provided you a table with project requirements, you must look at the provided table and project requirements and then pull out the most relevant item from the table.
"""

############

RULES_PROMPT: str = """Your job is to find the most relevant products in the table and returned it to the user.
But the matched product can be modified to FOLLOW THESE RULES: 
- If the floor has more than 100m x 100m area, you must use at least 4 cameras.  
- The project must always has onsite technical support at least one day
- The project must always has onsite technical training at least one day
- The project must always need account login on viAct.ai
"""

############

FORMAT_INSTRUCTION: str = """The returneded result must follow this CSV format: 
``` 
product\tproduct_id\tdescriptions\tprice\tpricing_unit\tcategory 
```
For example: 
``` 
product\tproduct_id\tdescriptions\tprice\tpricing_unit\tcategory 
Customized AI engine\tD011\tUsage of customized AI detection module (after Item 10.0 activated)\t$1,000.00\t/month\tDetection Modules
```"""

############

ASK_FOR_PROJECT_REQUIREMENTS_PROMPT="what equpiment do I need for this project? Give me a bulletin list" 