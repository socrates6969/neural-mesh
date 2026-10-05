"""R65b FRESH blind novel-phrasing routing test set (authored after R50b, WITHOUT looking at model errors on it; 10 per class)."""
ROUTE4 = {
"calc": ["What is 64 divided by 4 plus 9?","Please add 305 and 118.","Tell me 7 multiplied by 13.","What do you get when you take 500 and subtract 123?","Work out (8+2)*(3+4).","Find the value of 2^8 minus 100.",
 "How much is 99 plus 101 minus 50?","Compute 18 times 5 divided by 3.","What is the result of 1000 divided by 8?","Sum 11, 22, 33 and 44 for me."],
"units": ["How many millimetres is 15 centimetres?","Convert 3 kilometres to metres.","What is 2 hours expressed in seconds?","Tell me how many ounces are in 3 pounds.","Express 90 kilograms in tonnes.","How many weeks are there in 28 days?",
 "Convert 5 gallons into litres.","What is 12 inches in centimetres?","How many metres make up 3 miles?","Change 750 grams into kilograms."],
"csp": ["Place four items P, Q, R, S in distinct slots 1 to 4 where P comes before Q and R comes before S.","Three kids line up one behind another; Tim is ahead of Una and Una is ahead of Vic. Give the order.",
 "Find distinct finishing places for A, B and C such that B beats C and A beats B.","Seat Max, Ned and Oz in different chairs 1-3 so that Ned is left of Oz.","Rank five cars with no ties: car1 is faster than car2 and car2 is faster than car3.",
 "Assign unique positions 1..3 to the tasks e, f, g with e before f.","Order six teams without ties given that Alpha beats Beta and Gamma beats Alpha.","Which place does each runner take if Lia finishes ahead of Moe and Moe ahead of Nat?",
 "Distinct slots 1-4 for w, x, y, z where x precedes y and z precedes w.","Arrange three speakers in separate time slots so that speaker one is before speaker two."],
"sandbox": ["Reverse the digits list [4,5,6,7].","How many even values does [2,4,5,8,9,10] contain?","Sort [12,3,9,1] from smallest to largest.","Give me the last element of [3,6,9,12].","What is the sum of [10,20,30,40]?",
 "Rotate the list [1,2,3,4,5] to the left by two.","Return the minimum of 8, 3, 15, 6.","Drop the duplicates from [5,5,6,7,7,8].","Begin with 1 and double it six times; what do you end up with?","Add three to each number in [1,5,9]."],
"semantic": ["Choose the word closest to 'enormous': tiny, huge, fast.","Which one does not belong: car, bus, train, sandwich?","Fill in: The baby ___ when she was hungry. Options: cried, danced, evaporated.",
 "Hand is to arm as foot is to what? Options: leg, hat, cloud.","How are a violin and a cello alike? Options: instruments, vegetables, rivers.","Pick the antonym of 'ancient': modern, old, grey.",
 "Odd one out: Paris, London, Rome, banana.","Complete the sentence: The sun ___ in the east. Options: rises, melts, sings.","Teacher is to school as doctor is to what? Options: hospital, forest, river.","Which word is a synonym of 'tired'? Options: exhausted, energetic, loud."],
"knowledge": ["Who painted Starry Night?","What is the capital of Japan?","What is DNA?","Explain how a rainbow forms.","Why do we have leap years?","Which planet is closest to the sun?",
 "Who was the first person to walk on the moon?","What causes tides?","What is the boiling point of water at sea level?","Tell me about the Industrial Revolution."]}
