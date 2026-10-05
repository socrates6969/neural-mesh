"""R51b exemplar-augmentation set (authored separately from ROUTE2; used only as EXTRA kNN training exemplars, never for scoring)."""
ROUTE3 = {
"calc": ["What's 48 plus 77?","Give me 9 times 16.","Evaluate 1000 minus 357.","What is 250 divided by 5?","Compute 6*7+8.","Work out the sum of 120 and 45.",
 "What is 13 squared?","Find 15 minus 28.","How much is 72 divided by 8 plus 3?","Calculate 5^3.","What do I get from 31 times 3?","Add up 14, 26 and 40."],
"units": ["How many inches are in 4 feet?","Convert 10 pounds to kilograms.","What's 3 hours in minutes?","Express 2 miles in metres.","How many grams are in 1.5 kilograms?","Turn 8 litres into millilitres.",
 "How many days are in 5 weeks?","Convert 60 km to miles.","What is 500 mm in centimetres?","How many seconds make 7 minutes?","Change 12 yards to feet.","How many tonnes is 4500 kg?"],
"csp": ["Arrange Zed, Yan and Xia in distinct places 1-3 with Zed before Yan.","Seat five guests in different chairs so that Pia is ahead of Quin and Quin ahead of Ray.",
 "Rank three bands without ties: North beats South, South beats East.","Put tasks P, Q, R into steps 1 to 3 with P first and R after Q.","Four cyclists finish in distinct positions; Gil is ahead of Hal and Ivy ahead of Gil.",
 "Order A, B, C, D in unique slots where A precedes C and B precedes D.","Three students line up in different spots: Dan ahead of Eve, Eve ahead of Fay.","Distinct places 1..4 for W, X, Y, Z given W is ahead of X and Y is ahead of Z.",
 "Schedule three talks in separate time slots so that talk 1 comes before talk 3.","Five horses finish in unique places; Blaze beats Comet, Comet beats Dash.","Assign distinct ranks 1-3 so that J is above K and K above L.","Who finishes where if M beats N and O beats M among three runners?"],
"sandbox": ["Reverse the list [7,8,9].","How many odd numbers are in 1,2,3,4,5?","Sort [9,2,7,4] ascending.","Give me the second element of [5,6,7].","Total of the list [1,2,3,4,5].","Shift [1,2,3] left by one.",
 "What is the largest of 4, 18, 9?","Remove repeated values from [2,2,3,3,3,4].","Start at 2 and triple it four times.","Return [1,2,3] reversed.","How many numbers in [4,9,15,21] are over ten?","Increase every entry of [5,6] by two."],
"semantic": ["Which word is closest to 'happy'? Options: glad, sad, tall.","Pick the odd one out: rose, tulip, chair, daisy.","Complete the sentence: She ___ the door. Options: opened, flew, ate.",
 "Bird is to nest as bee is to what? Options: hive, tree, sky.","What do a cat and a dog have in common? Options: animals, tools, colours.","Choose the antonym of 'hot': cold, warm, big.",
 "Which does not belong: red, blue, apple, green?","Fill in the blank: The ___ barked loudly. Options: dog, cloud, spoon.","Pen is to writer as brush is to what? Options: painter, baker, pilot.",
 "Select the synonym for 'begin': start, end, wait.","In what way are a car and a bus alike? Options: vehicles, fruits, planets.","Find the word nearest to 'angry': furious, calm, green."],
"knowledge": ["Who invented the telephone?","What is the capital of Canada?","What is gravity?","Tell me about the Roman Empire.","Explain how a battery works.","Why do seasons change?",
 "Which ocean is the largest?","What causes earthquakes?","Who wrote Pride and Prejudice?","What does CPU stand for?","How do airplanes fly?","Describe how photosynthesis works."]}
