"""holographic_learnguard_examples.py -- the EXAMPLES the typed learning guard is built from (E4.2, 2026-09-26).

Data only: the typed guard (holographic_learnguard.SemanticGuard) derives its whole model from these lists at
runtime -- a deterministic encoder, one ProtoStore row per option below, one InfoNCE pass -- so no dense weights
are shipped anywhere. What a mind LEARNS on top (pattern refusals, learn_guard_example corrections) lives in its
own partition, never here.

GUARD_EXAMPLES = {option: (type, "example | example | ...")}. The option is "<source>:<intent>" and becomes one
prototype row; the type (credential / live_value / unclear / ordinary) is the row's KEY, so rows of the same type
never push each other (holographic_protostore.ProtoStore.key_of).

WHERE EVERY LINE COMES FROM (and why none of it can leak into the numbers it is judged by)
  * the TRAIN half of the candidate pool in tools/bench_learnguard.py (candidate_pool / in_heldout): the pool was
    written and selected BEFORE any tuning, then split by sha256 of each question. The other half is
    tests/data/guard_heldout.json and is never used here -- tests/test_learnguard.py asserts the two are disjoint.
  * up to 30 more CLINC150 / Banking77 examples per intent that were never in the pool at all (the pool took the
    first 60 per intent in a sha256 order; these are the next 30 of that same order). MEASURED on the train half
    (design harness, 5-fold CV, 0 false positives on realistic ordinary answers): live readings caught 226/286
    without them, 245/286 with 30 per intent, 244/286 with 90 -- so 30.
  * hand-written extras (options "hand:*:x_*"), written AFTER the split on 2026-09-26 by worker D-guard, after
    looking at train-CV misses (never at the held-out half or sets B/C). Mechanically filtered: 7 drafts within
    difflib ratio 0.8 of a held-out item and 2 within 0.8 of a B/C item were dropped unread. They are the
    identifiers-with-bare-token-answers the credential threshold was pinned by, and more credential / live wording.
  * ENGINE_EXAMPLES (bottom of this file): a frozen snapshot of the engine's own catalog phrases, all ordinary.
  The sweep-180 built-in lists (CREDENTIAL_EXAMPLES / READING_EXAMPLES / NORMAL_EXAMPLES) stay in
  holographic_learnguard.py and join as options "s180:*".

LICENCES: CLINC150 -- Larson et al. 2019, "An Evaluation Dataset for Intent Classification and Out-of-Scope
Prediction", CC BY 3.0. Banking77 -- Casanueva et al. 2020, "Efficient Intent Detection with Dual Sentence
Encoders", CC BY 4.0. Hand-written lines: questions only -- no credential VALUE appears anywhere in this file.

REPRODUCE (needs the datasets, which are not vendored): `python tools/bench_learnguard.py --check-examples DATA`
confirms every clinc150:* / banking77:* option below is exactly the train half of the pool plus extra_examples(), in
order (30/30 on 2026-09-26); `--build-heldout DATA` rewrites the held-out file byte for byte (sha256 24ba7f5b...).
The hand-written options and ENGINE_EXAMPLES are text here: there is nothing to regenerate them from.
"""

# data block: one option per entry, the examples joined by ' | ' and wrapped on spaces
GUARD_EXAMPLES = {
    "banking77:activate_my_card": ("ordinary",
        "Assist me please with card activation. | If you to to account, hit activate, and follow the "
        "instructions, you can activate it in just a few seconds. | Is my card ready for use or does it need "
        "activated and if so how? | What is the process of card activation? | What do I do to activate my new "
        "card? | What do I do to activate? | Help me activate my card. | I received my new card, how is it "
        "activated? | Can I activate my card? | What do I need to do to activate my new card? | I need to actuate "
        "my card. | I tried activating my plug-in and it didn't piece of work | How can I activate my card so I "
        "can start using it? | How can I activate my card> | How do I activate my new card I just got? | i cannot "
        "seem to activate card | Would it be possible to activate my card? | Would you be so kind as to activate "
        "my card? Thanks | How can my new card be renewed? | I want to start using my card, how do I activate it? "
        "| What are the steps to activating a new card? | How do i activate my card | How do I activate my card, "
        "so that I can start using it? | What am I going to need in order to activate my card? | I have my card "
        "now how do I activate it? | I would like to activate my card what do I need to do? | How can my new card "
        "be activated? | I want to use my card, how would I activate it? | Do I need a photo ID to activate a my "
        "new card? | Who do I call to activate my new card? | I would like to activate my card to use now, can "
        "you help me? | I have a new card and I need to activate it. | Can you tell me how I go about activating "
        "a new card? | Could you please activate my card | I just got my new card. How can I activate it? | I "
        "need info on activating my card? | will you be able to activate my card | Card activation is not "
        "working. What do i do? | I want to activate my new card. | I want to activate the card. | Please assist "
        "me in activating the card. | How can I switch on my new card? | How do I activate my card so I can start "
        "using it? | I couldn't complete card activation. | I would like my card activated. | What is the process "
        "for activating my card and using it? | My new card came in. How do I activate? | what do i need to have "
        "with me to activate card | Explain the activation method for this card | What is the activation process "
        "on my new card? | I can't activate my card? | I am planning activating my card was it possible? | My "
        "card activation is failing. | Can I use my new card? | Tell me what I need to do to activate my card. | "
        "Can I activate my card with the app? | Activate my card | The activation process for my card isn't "
        "working. | I am unable to activate my card, it won't let me."),
    "banking77:change_pin": ("ordinary",
        "I want to change my PIN - do I need to be in a bank? | I need to make my card PIN a different number | "
        "Can I change my pin through the app? | I think my pin has been compromised, what do I do? | Are their "
        "certain cash machines where I can change my pin? | Please help me set up a new PIN. | What do I need to "
        "do to change my PIN? | Help me change my PIN. | I want to choose a new PIN. | Can I change my pin number "
        "at a cash machine? | Can I change my pin at an atm? | Tell me how to setup a new PIN. | Can I change my "
        "pin at ATMs? If so, what ones? | Can I pick a new PIN? | Please help me change my PIN. | How can I "
        "change my Rowlock ? | I can't freeze my account as I need the card as I am traveling, how do I change my "
        "pin? | Show me how to change my pin? | Do I have to go into the bank to change my PIN? | I want to "
        "change my PIN. | Where can I change my PIN? | How do I change my PIN abroad? | I want to choose a "
        "different PIN. | Can my PIN be changed in any cash machine? | Is the bank the only place I can change my "
        "PIN> | Can I change my PIN on my phone? | I'm in Austria right now, but I really need to change my PIN "
        "right now. Can I do this even though I'm in another country? | If I am not in the country and I need to "
        "change my PIN, how can this be done? | Can i change my PIN at the ATM? | What kind of cash machines "
        "would allow me to change my PIN? | Do I have to change my PIN at a bank? | I am in Austria right now. I "
        "need to change my PIN ASAP, so can I still do this from here? | I'm travelling abroad but I've run into "
        "a situation where I need to change my PIN immediately. Can I do this from here? | I want to set a new "
        "PIN | Can I change my in at all ATM's? | Where do I need to go to change my PIN? | Is there a location "
        "where I can change my PIN? | Is there a location near me that i can change my PIN? | If I wanted to "
        "change my PIN, how would I do that? | Am I allowed to change my PIN anywhere? | I am travelling for two "
        "more weeks but really need to change my PIN asap. What do I do? | At what cash machines can I change my "
        "PIN? | How to change the PIN on my card? | Is there a way to change my PIN without having to go to the "
        "bank? | I want to create a new pin. | I need to change my card PIN. | In what way can I change my PIN "
        "and where do I need to be? | Can I change my PIN at any ATM? | I'm on vacation in Europe but I "
        "desperately need to change my PIN. Can I do this from abroad? | I am on vacation in Spain and think "
        "someone saw my pin when… Can I change it at a local ATM? | Your card pin can be changed at any Visa or "
        "MasterCard ATM with Pin services, excluding countries such as Belgium, Luxembourg, Austria, Germany, "
        "Spain, and France. | Can I change my PIN? | What steps do I need to take to change my card PIN? | Is "
        "going to a bank only way to change my PIN? | Can I change my PIN in Austria? | Can you tell me what I "
        "need to do to have my pin changed? | Please tell me how to change my pin. | Are there cash machines "
        "where I can change my PIN? | I want a new PIN please. | Do I need to go to a physical bank to change my "
        "PIN? | Can I change my card PIN? | Please let me know how to change my PIN? | What ATMs will allow me to "
        "change my PIN? | What do I have to do to change my pin?"),
    "banking77:compromised_card": ("ordinary",
        "My card has been compromised. I see a bunch of online shopping charges that I didn't make. I need to "
        "freeze it immediately. | Someone used my card without my permission. | How can I tell if someone else is "
        "using my card? | There's a possibility that my card was compromised. | How can I stop fraud on my "
        "account right now? | There are strange transactions on my account. What should I do if I think someone "
        "stole my details? | my card details could be stolen, as i didnt make these transactions | What should I "
        "do if I think someone is using my card without my permission? | There are charges on my card that I "
        "haven't purchased. | What do I do if I think someone managed to get my card information? | I think that "
        "someone may be using my card, but I'm not sure,. | What should I do if I think that someone else may be "
        "using my card. | Please tell me how you can stop unauthorized payments from being made on my card since "
        "on this bill I see that this has happened. There are transactions that I never made from a place I've "
        "never been to. | I think my card has been used by somebody else since I was never in the little town "
        "where some transactions on this bill have come from. Please stop this right away. | am suspicious of my "
        "card's security | What should I do if someone else used my card? | My card data has been exposed. | What "
        "should I do if I think that someone might be using me card? | There are transactions that I don't "
        "recognize. What do I do if I think someone is using my card? | My card was stolen and used to make "
        "several purchases. Please freeze my card so no one can use it. | Can you check about unauthorized use of "
        "my card, I think someone is using mine without my knowledge? | I think my child used my card while I "
        "wasn't home. | How do I stop fraud to my account? | How do I freeze my card? I think someone is using it "
        "to make a bunch of online transactions. | There are transactions that I did not make on my account, I "
        "think someone has my information. | Can I use app to freeze my card and dispute fraud? | My statement "
        "shows charges for things I never purchased. Were my account details stolen? | My card info was stolen, "
        "what do I do? | I have transactions that I don't recognize - I think someone is using my card. | How do "
        "I freeze my account? | I think someone may be using my card. | It seems someone used my card! There are "
        "a few transactions from a small town in the middle of nowhere that I definitely have not made! Please "
        "prevent them from using it immediately! | What do I do if I think someone has used my card without "
        "permission? I can see a few transactions I don't recognize. | I am not sure but someone else might be "
        "using my card | My card was used without my permission. | What do I do if someone used my card without "
        "my permission? | I think someone else is using my card. | I think someone is using my card without my "
        "permission! | Can I freeze my card right now? | My card was used without my permission, what do I do? | "
        "There are transactions that I don't remember making, i think someone might have gotten my card details "
        "and is now using it. | i didnt make these transactions, i think my card details might be stolen | How do "
        "I report possible fraudulent activity on my account? | I think someone has hacked my card! | If I feel "
        "someone has my card information, can I get a new card? | What can I do if my card details where stolen "
        "from my car? I think they used my card to buy gas. | I might be paranoid, but someone may be using my "
        "card. | It looks like someone besides me ordered something with my card, what should I do? | There is "
        "unusual activity on my account and I believe someone took my card. | I believe that someone is using my "
        "card without my knowledge! | I think someone stole my card number and brought stuff in different places "
        "that I have never been to. I need to suspend any further purchase on my credit card. | What do I do if I "
        "think someone has used my card? | am afraid someone may have gained access to the info on my card. | I "
        "can't be sure, but I think that someone may be using my card. | I think someone has access to my card "
        "numbers that shouldn't. | Someone might be using my card. What should I do? | Someone hacked into my "
        "cards account."),
    "banking77:get_physical_card": ("ordinary",
        "How do I locate my PIN now that I have my card? | What do I do with my card PIN? | I am yet to receive "
        "my pin. | Is my PIN sent separably? | Where in the app can I find my PIN? | So, the card PIN? | I am not "
        "able to see the card PIN anywhere? | Where is the PIN number found? | I need my PIN, where is it? | "
        "Where do i find my PIN? | How do I get a PIN? | Where can I locate my PIN at? | Explain the card PIN to "
        "me. | How do I get started when I get my card? | Are PIN separately? | Will my pin number be sent with "
        "my card? | my pin hasn't arrived in the post! How do I cancel it or get a new one? | Where is my pin? I "
        "don't have it yet | still need to get card pin | What can you tell me about the card PIN? | Will the PIN "
        "come separately? | Where is the card PIN? | How do I set up my card PIN? | Is my PIN the same thing as "
        "my passcode? | what is the process for setting up a pin | How do I find my PIN number? | Show me where I "
        "can see the PIN? | The card PIN? | I have not received my pin yet | How do I find my card PIN? | Where "
        "is my PIN number located? | Can you deliver the PIN separately? | do i have to wait for a physical card "
        "before i get my pin | how to get card pin? | I haven't received the PIN yet. | Where can I find my card "
        "PIN? | I havn't received my PIN yet. | When do I set up my card PIN? | Is my PIN recorded anywhere? | I "
        "haven't received my PIN yet. Do I need to get it from you? | Where is the PIN for my card located? | "
        "Does my PIN come with my card? | Where's my card PIN? | I have a question regarding the PIN of the card? "
        "| How is my PIN sent? | Where should I look for my PIN number at? | can i create my own pin right away | "
        "I'm looking in the App and can't find my PIN, where should I look? | In regards to the PIN of the card? "
        "| I cannot see the card PIN anywhere? | I signed up, but don't have a PIN. Where can I find it? | i "
        "cannot find my card PIN | Where is my PIN located? | I do not have my pin | How do I find the PIN? | I "
        "can't find my PIN anywhere. | The card PIN is not visible anywhere? | I cannot locate the card PIN. | "
        "Please help in finding my card PIN, thank you! | How do I set-up my PIN for the new card?"),
    "banking77:passcode_forgotten": ("ordinary",
        "What do I do if I have forgotten my passcode? | How do I change my passcode? | Where can I get a new "
        "passcode? | Where do I go to reset my passcode? | What do I do if I can't access my passcode? | "
        "Something is wrong with my password. | I can't get into the app with the passcode. | I think my passcode "
        "was changed | Are there directions to get a new passcode if I forgot mine? | I need a new login code | I "
        "don't know what my passcode is, can you help? | I forgot my code to get into the app. | I don't remember "
        "my password | Can someone help with my passcode? | Can I be given a new passcode? | What is the code I "
        "need to get into the app? | I seem to of forgotten my passcode. | I can't access my account. | I forgot "
        "my password to get into the app! | How can I reset the passcode if I need to do that? | I have forgotten "
        "my passcode | I don't remember my code to get into the app. | Lost password | I don't know the code for "
        "the app. | Can you tell me what to do to reset my passcode? | Can my password be reset if I do not have "
        "it? | I want to reset my passcode, how can I do that? | I don't remember my passcode? | What can I do if "
        "my passcode won't work? | I need to reset my passcode, this one isn't working | My passcode won't work. "
        "| Tell me how to reset the passcode. | I don't have my passcode to access the app. | How do I get the "
        "passcode reset? | I don't know my password anymore. | What do I do if I forget my passcode? Because I "
        "did. | I am entering my passcode but getting an error. | what is going on, i have entered my passcode "
        "and its not working | I happened to forget my passcode | I can't remember my password. | For some reason "
        "I forgot the passcode I have. | Can I resent my passcode? | I can't enter my passcode. | Can I reset the "
        "passcode? | Help me. The passcode doesn't work. | I think I forgot my passcode | I forgot my passcode. "
        "Now what? | My password isn't being accepted and I need to reset it. | Why won't my passcode work? | I "
        "lost the code and can not get into the app. Help! | How do I retrieve my passcode? | How can I change my "
        "password? | I have forgotten my password. | Can you help me reset my passcode? I forgot it. | I no "
        "longer have my passcode. | I can't recall my passcode and need to reset it. | I do not know my passcode. "
        "| I do not understand why my pass-code is not working. | I've forgotten my passcode. Can I reset? | Can "
        "you help me reset my password? | I don't remember my login code | Disaster, I've totally forgotten my "
        "passcode, can you help me? | I need to reset the passcode. | I lost my passcode."),
    "banking77:pin_blocked": ("ordinary",
        "Where can I view my PIN? | Will you reinstate my PIN? | Why is my PIN blocked? | How to unblock my PIN? "
        "| Can you unblock my blocked pin? | Where do I go to unblock my PIN? | I attempted to enter my pin to "
        "many times. | What is the procedure of unblocking my PIN? | How do I unblock my PIN? | My PIN has been "
        "blocked, what should I do? | I accidentally exceeded my PIN tries. Please advise. | My PIN number was "
        "incorrect, and I can't access it. | I need my PIN unlocked. | My PIN is not working and I need "
        "assistance. | Would you please unblock my pin? I don't know why, but it's blocked. | How do I unblock my "
        "card? | How do I reset my PIN? | I logged in wrong and am blocked, how do I log in? | My card got "
        "blocked, how do I reset? | What steps do I take to unblock my PIN? | My account is locked because I used "
        "the wrong pin too many times. Please help.p | What do I do if I have exceeded all my PIN tries? | Can I "
        "reactivate my PIN? | I accidentally blocked my PIN. How do I reset it? | I can't input my pin again. | "
        "Where can I go to get my PIN unblocked? | Can you assist me with unblocking my PIN? I put it in wrong "
        "too many times. | I have used all of my PIN tries. What should I do now? | My PIN was entered wrong and "
        "now I am blocked. Please unblock. | How do I get unblocked? | I managed to drunken block my card :(( "
        "Help! | Can you unblock my account? I entered the PIN wrong. | What if I type in the wrong PIN too many "
        "times? | I may have entered the PIN wrong and the account is blocked. What do I have to do to get it "
        "unblocked. | How do I deal with a blocked PIN? | Help! I forgot my PIN and have been locked out of using "
        "my card. | How do I get my PIN unlocked? | How can I unblock a blocked pin number for my account? | My "
        "PIN can't access my card, can you help?. | I forgot my PIN and now it is blocked. | Help me unblock my "
        "PIN. | I used the wrong ping too many times and now the account is blocked. How do I unblock? | I am "
        "locked out from entering my pin. | I blocked my card by mistake, how do I unblock it? | After putting in "
        "the wrong PIN too many times, I was blocked. Can you assist me in changing it? | How many times can I "
        "enter a wrong PIN before it is blocked? | I attempted to use my card while I was intoxicated, and I "
        "failed to input my PIN, and the machine kept my card. How soon can I have it back? | How many incorrect "
        "attempts cause card to be blocked? | Can you unlock my pin? I think I entered the wrong pin too many "
        "times. | I can't figure out how to unblock my pin number. Can you help me? | What do I have to do to get "
        "my PIN unblocked? | After inputting the wrong pin too many times, can you now help me unblock my pin? | "
        "Can I unblock my pin? | My PIN is not working, can you help? | My pins seems to be blocked, can you "
        "unblock it please | My account is blocked, how do I log in now | I have exceeded all my PIN tries"),
    "clinc150:balance": ("live_value",
        "will you let me know my bank balance | what is in my bank accounts | what's my bank balance | what are "
        "my coffers at | do i have any cash left | how much is the current balance in my td bank savings account "
        "| i need to know how much money i have in all of my bank accounts | do i have enough money in my first "
        "hawaiian bank account for a vacation | i'd like to know my bank balance please | what is my current "
        "balance on my home equity line of credit | do i have enough in my chase account for a plane ticket | "
        "check my visa account and see if i have enough money for dinner tonight | what is is the details of my "
        "bank account | how much money is in all of my bank accounts | how much do i have in my bank accounts | "
        "is there any money left | how much money do i have in my pnc account | do i have enough in my chase "
        "account for new nikes | perform a search for my most recent balance on my amex account | what's the "
        "balance of my savings | what's the balance in my checking | check chase bank for my checking balance | "
        "how much money is there in my bank accounts | what do i have in my bank accounts right now | can you "
        "tell me how much money i have i my bank accounts | please find my balance on my chase mastercard | can "
        "you tell me my checking account balance | what is my checking account balance at chase | what is the "
        "available balance in savings | how much money do i have total | what's my savings balance at chase | "
        "what is my balance | what is the balance on my visa | what is my bank balance | will the amount in my "
        "chase bank account right now cover the cost of a new dryer | how much is in my pnc account | what is my "
        "saving's account balance | what's the balance on my bank account | do i have enough in my wells fargo "
        "account to get some nike's | what what kind money is available in my bank accounts | do i have enough "
        "money in my charles schwab account to get a new baseball bat | what's the balance of my bank accounts | "
        "what's my pnc balance | do i have enough money in my chime bank account to take ashley to the movies "
        "tuesday | will the money in my capital one account cover a new washing machine | could be there be a "
        "good amount of money in my checking account to go on a vacation | what is my savings account balance | "
        "can you tell me my bank balance | how much dough do i have in my bank accounts | what is my bank balance "
        "for all accounts | what amount of money is in my bank accounts | can i get beer within my deposit "
        "account | do i have money in my wells fargo account for nike's | savings account balance at chase bank "
        "please | how much moola is in my bank accounts | what is the balance of my bank of american account | "
        "what's my total net worth in all of my bank accounts | i need to know my bank balance | what's my "
        "checking balance | check my wells fargo account to see if i have enough for these nike's"),
    "clinc150:calculator": ("ordinary",
        "what is 80980 + 098098 + 80980 + 1243 | what is 55 times 300 | what is 100 multiplied by 55 | what is "
        "1785 minus 334 minus 87 plus 374 minus 400 plus 17 | can you help me solve a math problem | tell me what "
        "1875 plus 3459 equals | what is 2/3 x 1/9 | i bought 6 shirts at $499 each what was my total expenditure "
        "for them | how many times does 8 go into 2000 | what is 300 divided by 42 | find the square root of "
        "1243435 | how many times can 12 go into 600 | i need you to help me with some math if you can | what is "
        "the sum of 3 plus 5 | please help with my math | what is the antilog of 365 | if i win 200000 how do i "
        "split it 7 ways | what is 2 + 2 | help me solve this math equation | can you tell me what 30% off 235 "
        "is, please | can you do algebra | what is 543 times 344 | can you calculate 18 divided by 45 | what is "
        "63 percent of 145 | what is 10 to the 12th power | what's 47 times 83 | can you tell me what 30% off 235 "
        "is | i need to know what 75 plus 43 is | how to solve this math problem | what is 20+ 5 | what is 87 "
        "divided by 4 | what is the area of a 20 x 20 room | i need to know what 25 times 38 is | subtract 85 "
        "from 997 | what's 3 plus 3 | what is 250 times 118 times 9 | what is 562 times 400 | what is 592 minus "
        "124 | what is the square root of 66 | what is the solution to sixty times thirty | please add 456 and "
        "781 for me | what is six divided by 16 | what is the square root of 1 million | what is 1243 times 45 | "
        "what is 4 x 4 | what is 1 million twelve hundred divided by 400 thousand | can you tell me what 80 "
        "divided buy 4 is | add up 8 and 7 | can you factor x squared plus 4 x plus 4 | can you tell me what 30% "
        "off 279 is, please | what is 213 times 3 | what is 750 divided by 5 | what is the square root of 888 | "
        "what is 10 + 10 | what is 20 times 20 times 30"),
    "clinc150:calories": ("ordinary",
        "how many calories are in a can of coke | i need to know the calorie content of spaghetti | what's the "
        "calorie info for 2 cups of regular chex mix | how many calories on average are in a hot dog | are there "
        "a lot of calories in muffins | in one cookie, how many calories would i find | what type of calorie "
        "numbers are in onions | how many calories are in a cookie | how many calories does a scoop of chocolate "
        "ice cream contain | i need to know the calorie content in a piece of pepperoni pizza | what are the "
        "calories in a cookie | the intake of calories it it bad | how many calories in scoop of chocolate ice "
        "cream | how many calories does two bananas have | what is the calorie content in spaghetti | how many "
        "calories in meatloaf | how many calories are in a cheeseburger | what is the calorie going to be if i'm "
        "eating cereal | i need to know the calorie count of a bacon cheeseburger | whats the calorie content of "
        "oatmeal | what's the expected calories in a cream filled cookie | what is the calorie content in peanut "
        "butter | how many calories are in cake | how many calories are in a bowel of wheaties | what is the "
        "number of calories in a steak | what's the calorie content of cheetos | what's the expected calorie load "
        "of a peanut butter and jelly sandwich | calorie check, cheese burger | how many calories are in mashed "
        "potatoes | do cheetos have a lot of calories | can you tell me the number of calories in one serving of "
        "whole cashews | what is the amount of calories that scrambled eggs has | what amount of calories are in "
        "one muffin | what is the calorie content in potatoes | how many calories are in cookies | how many "
        "calories does a big mac have | what's the caloric content of an apple | one cup of almonds has how many "
        "calories | what's a big bowl of ice cream contain in calories | how many calories are in a piece of "
        "bacon | what's the calorie content of chicken nuggets | how many calories are in a sandwich | do you "
        "know the calorie content of a strip of bacon | what is the calorie content of lays chips | please tell "
        "me how many calories one chocolate bar contains | how many calories would i consume if i ate a loaded "
        "hotdog | how many calories are in oreos | how many calories are in powdered donuts | number of calories "
        "in coke | add up the calories in tacos | please tell me the total calories a single serving of chocolate "
        "ice cream is expected to contain | what is the calorie content in french fries | what's the calorie "
        "content of french fries | what's the caloric content in a bowl of rice crispies with milk | can you "
        "check the amount of calories in a chicken sandwich | how many calories are in waffles | how many "
        "calories are in a burger | how many calories in peanuts | a bowl of cheerios with milk has how many "
        "calories | do you know how many calories are in a single chicken breast | what is the calorie content in "
        "bananas | how many calories does an orange have"),
    "clinc150:cook_time": ("ordinary",
        "how long should i wait before i can bake bread with homemade dough | how long should i cook country "
        "fried potatoes for | what do i set the timer for if i'm making gyoza | how long will it take to fix "
        "shepherd's pie | how long should i cook my turkey | how long will it take to cook a lasagna | how long "
        "do i cook chicken breast | how long should i cook chicken | how long should i cook a steak for | how "
        "long do i cook the chicken roast | when can i expect meal of salmon to be finished | how long does it "
        "take to cook steaks | how long to microwave a frozen dinner | how long does it take to bake a cake | if "
        "i cook the pizza at 400 degrees how long must it be in for | how long do i put the casserole in | how "
        "long do cheeseburgers take to make | how long to cook steak for | how long does it take to boil an egg | "
        "how long do i need to cook chicken | tell me how long i will need to spend preparing a meat loaf dish | "
        "how long do i need to cook lasagne for | what's the cooking time for chicken alfredo | for how long "
        "should i bake the brownies | how long should a cake bake | how long do noodles need to cook | how long "
        "do you cook pasta for | how long does it take to prepare pot roast | how long do you cook a salmon | how "
        "many minutes do i cook baked ziti | how long should i boil eggs | how long do i cook this for | i need "
        "the cooking length for a turkey | how long should i cook a glazed ham | how long should i cook the eggs "
        "for | how long does it take to cook ham | how long do i need to cook roast for | how long do you need to "
        "put the cuish in for | how long should i cook carnitas for | so how long do you think the chicken will "
        "take | what's the recommended cooking time for steak | how long should i bake the brownies for | how "
        "long should i cook pho for | about how long should you cook lasagna | how long should i cook ham for | "
        "how long do i cook pork | how long will it take to make an omelet | how long do i boil eggs | how long "
        "does it take to cook meal of tuscan | how long should i cook the ham for | how long should i cook "
        "chicken thighs | how long must i cook spaghetti for | how long do i need to cook chili for | how long "
        "should i expect beef stroganoff to prepare | what's the time it takes to make a decent omelette | how "
        "long does a roast take | how long am i supposed to cook pork loin | the sauce must simmer then go in the "
        "oven but for how long | how long does it take to cook hot pockets | how many minutes should i set an "
        "alarm for this bake"),
    "clinc150:credit_limit": ("ordinary",
        "tell me the highest amount i can spend on my discover card | what is my spending limit on my chase "
        "sapphire card | what is my visa card limit | i would like to know the credit limit for my citibank card "
        "| what's my credit limit | how is that credit limit | how high is my credit limit for my american "
        "express card | what is the credit limit for my bank of the west card | how much can i spend on my visa "
        "card | my visa has what limit | what is the spending limit on my visa | what is the limit on my "
        "victoria's secret card | what's the limit for my credit | tell me the limit for credit on my mastercard "
        "| what's my limit on my visa card | do you know my credit limit for the visa card | i want to know my "
        "credit limit | how much can i charge on my visa | what's my credit limit on my visa | what is the credit "
        "limit for my navy federal card | what is my amex credit limit | how much do i have available on my visa "
        "card | i wish to know my credit limit for my credit card | can you find my credit limit on my mastercard "
        "| could you share what my current credit limit is | report my credit limit | i need to know what my "
        "limit is on my american express card | calculate the limit i have available for spending on my natwest "
        "card | what credit limit do i have | is there a spending limit | how much is my credit limit | how much "
        "to i have left on my visa card limit | how high is my credit limit for my wells fargo card | how high of "
        "a credit limit do i have on my amex card | explain what my credit limit is | i want to know my spending "
        "limit | what is my credit limit on my discover card | tell me my hsbc card credit limit | do i have a "
        "monthly spending limit | can you tell me the limit i currently have on my barclay card | i have to know "
        "the credit limit | how much is my maximum to spend on credit cards | how high is my credit limit for my "
        "bank of america card | i would like to know the limit for my credit | what's my spending limit on my "
        "discover | credit limit info | how much can i spend on my discover card | what is my spending limit on "
        "my mastercard | please check my credit limit on my visa | could you tell me the limit on my chase credit "
        "card | what's the credit limit on my discovery card | i need to know what the visa card credit limit is "
        "| what is the current credit limit i have on my wells fargo mastercard | what is my credit max | how "
        "high is my credit limit | what is the credit limit on my visa | what's my credit limit for my visa card "
        "| can you tell me what the limit is on my capital one card | how much can i spend wth my visa | can you "
        "let me know what my credit limit is for my visa card"),
    "clinc150:credit_score": ("ordinary",
        "help me locate my credit score | i want to find out what my credit score is | how do i locate my current "
        "credit score | i wish to know my credit score | i would like to know my credit score | tell me my credit "
        "raing | are you able to lookup my credit rating | how is my credit score rated | provide me with my "
        "credit score | how do i see my credit score | how is my credit score numberwise | what is the process of "
        "finding my credit score | any idea what my credit score is | i wish to know my credit rating | let me "
        "know what my credit score is | how to see my credit score | can you help me find my credit score | look "
        "up my fico credit score | can you reveal my credit score | how do i get my credit score | give me my "
        "credit score | please lookup my experion credit score | what is my current credit score | tell me my "
        "current credit rating | how do i find information about my credit score | how high is my credit score | "
        "find my credit score | whats my credit rating | what number is my credit score currently | tell me my "
        "credit score | who tracks my credit score | if i want my credit score, how do i find it | how bad is my "
        "current credit score | how exactly do i find my credit score | i would like to be told about my credit "
        "rating | how is my credit score | show me my credit score | i wanna know my credit rating now | is my "
        "credit score over 700 yet | what is my credit rating | i would like to look up my credit score please | "
        "where can i check my credit score | where's my credit score | where is my credit score located | where "
        "can i see my credit score | can you check my credit score for me | how much is my credit score | i need "
        "to find my credit score | tell me what my credit rating is | where is my credit score | how do i locate "
        "my credit score | what's my credit score rating | what's my credit rating | how do i find my credit "
        "score | let me understand my credit rating | please look up my credit score | lets look up my credit "
        "score | do you know my credit rating number | i'd like the number for my credit score | please get my "
        "credit score | can you tell me my credit score | how can i request my credit score | could you tell me "
        "what my credit score is"),
    "clinc150:date": ("live_value",
        "what day is it | tell me what the date is today | what date is it today | what is the date of today | "
        "what's today | is it monday, tuesday, wednesday, thursday, friday, saturday, or sunday | what date will "
        "it be 9 days from now | what day of the month is it | i'd like to know the date tomorrow | what's the "
        "date 400 days from now | what day it today | please tell me what date it is today | let me know what "
        "today's date is | do you have the date | which day will it be five days from now | tell me the date it "
        "will be 5 days from now | today's date is what exactly | i'd like to know what the date is | in 100 days "
        "from now, what will be the date | give me tomorrow's date please | what would the date be 5 days from "
        "today | what day is it today | when will it be in 10 days | can you tell me tomorrow's date | what year "
        "is it | tell me what day today is | what is the date tomorrow | tell me what tomorrow's date is | what "
        "is the date for tomorrow | would you please tell me today's date | what date will it be tomorrow | after "
        "another eight days, what day will it be | i need to know what the date is today | i would like to know "
        "tomorrow's date | what date is today | could you tell me today's date | i need the full date for today | "
        "let me know what tomorrow's date is | what day of the month is it today | today's date is what | what "
        "date is it tomorrow | show me the date | let me know what date it will be in 3 days | can you remind me "
        "of the date | i need to know what date it is tomorrow | i would like to know what today's date is | in 4 "
        "days, what date will it be | what date will it be 7 days from now | what will the date be in 64 days | "
        "what is today's month, day and year | is today monday | what is the month and day tomorrow | i would "
        "like to get today's date | what day of the week are we on | what is tomorrow's date, please | what's the "
        "name of the day today"),
    "clinc150:definition": ("ordinary",
        "what does serendipity mean | define monetary for me please | what's the definition of luminescent | what "
        "does confrontation mean | what the heck is qat | what does epicurean mean | what does mature mean | "
        "definition of anachronism | what is the definition of flange | tell me the meaning of hegemony | what is "
        "an anachronism | what does amicable mean | what does altruism mean | can you tell me the definition of "
        "yttrbium | what does ferrari mean | the definition of affiliate is | what does adulation mean | what "
        "does \"money\" mean | what is meant by defense | what is the definition of incomprehensible | do you know "
        "what calumny means please look it up for me | look up zesty in dictionary | what does ajar mean | whats "
        "the definition of poor | can you show me the dictionary definition of ajar | i need to know what "
        "dominate means | define zesty | can you tell me what obsolescence means | tell me the definition of the "
        "word redemption | what is the meaning of the word lux | define antebellum | what does peer-to-peer "
        "actually mean | define qat for me | is \"rescind\" a word of positive connotation | what is the meaning of "
        "supercede | tell me please what paean means | what is the definition of didactic | give me the meaning "
        "of alternative | may i please have a definition for the work churlish | tell me what the word die means "
        "| flange means what | what does zesty mean | what is the meaning of word alliance | find what the word "
        "diaphanous mean please | tell me the meaning of condemnation | what is the meaning of "
        "interorganizational | what does the word ataraxy mean | what is regard mean | what is stupedous meant | "
        "tell me what the word bounty means | what does assiduous mean | define flange | tell me what monstrosity "
        "means | use the word ataraxy in a sentence | i want to know what trenchant means | define sonogram | "
        "what's the meaning of affiliate"),
    "clinc150:exchange_rate": ("live_value",
        "what's the difference between usd and cad | look up the rate of exchange between pesos and usd | i must "
        "know five dollars in yen and rubles | in canadian dollars, what is $30 | convert 100 dollars to euros | "
        "what is one dollar worth in mexico | how many japanese yen are in a us dollar | what is the rate for 500 "
        "cad in usd | tell me the current exchange rate between cad and euros | what is 5 in yen and rubles | how "
        "many us dollars can i get for 20 euros | whats rupees dollars in 30 | exchange rate to go from dollar to "
        "yen | how many pesos equals 500 dollars | how many dollars can i exchange for 10 yen | what's the "
        "exchange rate between dollars and pesos | look up the exchange rate between dollars and euros | what's "
        "100 dollars in euros | what is the exchange rate between euros and pesos | what is the exchange rate for "
        "canadian dollars to us dollars | i wanna know the exchange rate between yen and dollars | how many "
        "dollars can i exchange for 200 yen | how much is $30 usd in canadian dollar | how much is 500 dollars in "
        "pesos | how many pesos in one dollar us | if the dollar worth a lot in country b | what is the maximum "
        "dollars i can get for 6 yens | what's 10 euros in dollars | i would like to know the usd to aud "
        "conversion rate | how many mexican pesos can i get for one us dollar | us and mexico exchange rate | "
        "whats the current exchange rate between usd and eur | what is the amount of dollars i get if i trade in "
        "6 yens | how much is 10 us dollars in canadian dollars | what is the exchange rate from pounds sterling "
        "to us dollars | tell me the currency conversion rate from kurus to euros | can you please convert $30 "
        "usd to canadian dollars | what's the exchange rate betwen usd and euros | what's dollars yen in 10 | let "
        "me know the exchange rate between dollars and rubles | what is the quantity of dollars i receive for "
        "trading 6 yens | whats pesos australian dollars in 20 | i need to know how much 100 dollars is worth in "
        "euros | how much is the exchange between usd and euros | whats euros kroner in 25 | what's the "
        "conversion rate of 100 dollars to euros | what is the exchange rate between rubles and us dollars | how "
        "much country of canada money would i get for $100 | dollar to pesos exchange rate | how many mexican "
        "pesos is a us dollar worth | how many dollars can i exchange for 5000 rubles | how many dollars can i "
        "exchange for 75 euros | how many euros can i exchange for 5 us dollars | what is the exchange to yen if "
        "i have 100 us dollars | can you tell me what 100 british pounds equals in us dollars | convert "
        "krugerrands to saudi riyal | can you tell me today's rate for cad to usd"),
    "clinc150:expiration_date": ("ordinary",
        "where should i look for my credit card expiration day | where can i find my credit card's expiration "
        "date | what is the expiration date on my card | how long is the validity of my credit card | what is the "
        "expiration date on my visa card | what's the month that my card expires | could you look up the date "
        "that my credit card is set to expire | when is the expiration date for me discover | is my credit card "
        "expiring soon | how long do i've got until my discovery card expires | what is the expiration date of my "
        "wells fargo card | when is my credit due to expire | what month does my card stop working | how long is "
        "it going to be until my card expires | tell me the expiration date on my visa credit card please | "
        "what's the last date i can use my credit card | during what month will my card's expiration date fall on "
        "| when is my credit card going to expire | how many days until my amex card reaches it expiration date | "
        "will i need to renew my credit card soon | how long is it until my credit card expires | when does my "
        "amex expire | let me know when my credit card expire | tell me when my credit card expires | what is the "
        "month of my card's expiration date | when do i need a new credit card | when does my discover card "
        "expire | do you know the expiration date that is on my visa card | help me figure out when exactly my "
        "credit card will be expiring | i need to know when my card is set to expire please | does my credit card "
        "expire soon | my card will work until what month | how long until my mastercard expires | what is the "
        "expiration month and year on my credit car | when should i expect my visa to expire | i need to know "
        "when my mastercard will expire | when is my card expired | my discover card expires on what date | can "
        "you check when my visa card expires | when it my citi card expired | what's the exact date that my chase "
        "card will no longer work | when will my new credit card arrive | what month does my credit card expire | "
        "tell me the expiry date of my credit card | on what date does my visa card expires | how long before my "
        "amex card expires | when will my american express card expire | i want to know the expiry month of my "
        "card | when is my credit card set to expire | can you find the expiration date and tell me what it is "
        "that is on my visa credit card | will my card expire on a certain month | tell me the expiry month of my "
        "card | which month is the one in which my card expires | what is the month that my card is set to expire "
        "in | what month is my card expired in | when will my american express credit card expire | in what month "
        "will my card reach expiration | can you find out for me when my credit card will be expiring | do you "
        "know the expiration date on my credit card"),
    "clinc150:flight_status": ("live_value",
        "what time will i be able to board the plane | when is my flight scheduled to board | whats my delta "
        "flight's status | when's my flight boarding | give me the status on my united airlines flight | can you "
        "tell me the status of my delta flight | when should the flight land | at what time is my flight "
        "scheduled to land | what is the eta of my flight | tell me what time my flight ought to be landing | "
        "what's the status of my jetblue flight | what's the status of my american airlines flight | when is my "
        "flight supposed to leave | tell me when my flight scheduled to board | when will my jetblue flight get "
        "here | what time is my flight supposed to be landing | when is my flight expected to arrive | i need to "
        "know when my flight is landing | can you tell me when my flight is going to board | i would like to know "
        "when my flight scheduled to board | is my flight, dl123 on time | at what point should the flight land | "
        "what's the status of my virgin airlines flight | is flight dl123 going to arrive on time | when is "
        "boarding scheduled for my flight | i would love to know when my flight is going to land | when should i "
        "expect my flight to come in | is flight dl123 coming in on time | has flight dl123 left | at what time "
        "will my flight begin boarding | when is my flight boarding | what is the scheduled arrival time for my "
        "flight | tell me what the status of my american airlines flight is | i would like to know flight dl123's "
        "status | tell me when my flight is scheduled to start boarding | what's my flight's eta | please tell me "
        "the status of my southwest flight | i need to know the status of flight dl123 | what's the latest info "
        "on flight dl123 | when will we begin to board my scheduled flight | has flight dl123 arrived yet | when "
        "will my aa flight be landing | what is the arrival time for my flight | when is boarding scheduled | "
        "tell me the status of flight dl123 | can you update me with the status of flight dl123 | what time will "
        "i be allowed to board | tell me flight dl123's statud | will flight dl123 be on time | what time is my "
        "flight scheduled to leave | what's the latest with my united flight | update me on my delta flight "
        "please | i gotta know when my flight will land | do you mind informing me the status of my southwest "
        "flight | can you check when my flight lands | what is the status of my frontier flight | what's the "
        "status of my united flight | give me the status of flight dl123 | tell me the status of my american "
        "airlines flight | when is my plane scheduled to land | whats the latest on flight dl123 | could you tell "
        "me the status of flight dl123 | what time is this flight supposed to land"),
    "clinc150:fun_fact": ("ordinary",
        "what is the interesting trivia about human bodies | repeat that fun fact about mt everest | can you "
        "share something interesting about world war one | help me learn something intriguing about turtles | "
        "what is a fun fact about mt everest | give me a cool fact about seattle | tell me something interesting "
        "about lake tahoe | what's a fun fact about mythology | tell me some something trivia | list some neat "
        "stuff about rats | tell me something interesting about elephants | what's a fun fact about axolotls | "
        "give me a fun fact about dolphin | give me a fun fact about komodo dragon | i'd like to hear a fun fact "
        "about the nfl | tell me some trivia about dolphin | what is some trivia about frog | i want to learn "
        "something about apples | give me something interesting about stars | say trivia about lebron james | "
        "what are fun facts about lighthouses | lets fin out some fun facts about humans | tell me something "
        "interesting about sperm whales | tell me something interesting about bees | tell me all about the trivia "
        "with friends | read me some interesting information about cats | tell me fun facts on science topics | "
        "have any cool facts on reptiles | i want a fun fact about london | tell me something interesting about "
        "new york state | what is today's fun fact | tell me some trivia about birds | i want to know some trivia "
        "about our solar system | tell me a fact | what is the most unique piece of trivia relating to cameras | "
        "lets hear a fact | i want to learn a neat fact about black holes | what is one fun fact | give me a cool "
        "fact about lsd | tell me something cool about elephants | can you tell me a fun fact regarding cars | "
        "tell me something interesting about dogs | can you tell me something i don't know about banks | tell me "
        "a few facts about cats | give me a cool fact about potatoes | what's a good fun fact about great britain "
        "| read me some fun facts | tell me a fun trivia bit about artificial intelligence | do you know any fun "
        "facts about the ocean | do you know any trivia about the ocean | i need a spider fact | what would be a "
        "fun fact about cats | share with me a fun fact about outer space | give me a fun fact about kangaroo | "
        "i'd like to hear some trivia about florida | give me an interesting fact about antarctica | lets go "
        "threw some trivia on sports | i wanna hear something cool about bees | what's a cool and interesting "
        "thing about dogs"),
    "clinc150:how_busy": ("live_value",
        "can i expect chili's to be busy at 4:30 | what's the wait at macaroni grill | how much to wait before "
        "dining in the jack in the box at 4 pm | what is the average wait time to be seated for dinner at rouge | "
        "what's the table wait at applebees | how busy is zippy around 12 for lunch | i want to know how busy "
        "ruby tuesday will be at around 8:45 pm | would you say that red lobster's pretty buy at noon | how long "
        "is the wait at macaroni grill | around 5 pm, how busy is kaya | how busy is mc donald at 8 in the "
        "morning | can you tell me how long approximate wait times are at cheesecake factory | how busy will "
        "golden corral be at 7:30 tonight | how long is the wait at applebee's | how long will it take to get "
        "seated at needham's | i need to know about how much time i'd have to wait to get a table at dibruno's | "
        "would tio's be crowded at 7 | how busy is cheesecake factory right now | will i have to wait to get a "
        "table at ihop | i want a table at texas roadhouse; how long will it be | do you think longhorn "
        "steakhouse will be busy at 5pm | what's the wait for a table at olive garden right now | how many people "
        "go to chili's around 9pm | find out what wait times are like right now at olive garden | will cracker "
        "barrel be crowded around five this evening | how long is the wait at orchids | how long to be seated at "
        "carrabas | what is the wait like at apple bees | tell me how busy the restaurant will be between 5 and "
        "7pm | what is the wait supposed to be at zippys | how long would i have to wait if i want to go to "
        "golden corral | at 5:30 pm, how busy can i expect olive garden to be | tell me mr joes pizza average "
        "wait time | how busy is michel at 9 | how busy is shokudo at 12 | how long is the wait at buffalo wild "
        "wings | what's the typical time to eat at red lobster | how busy is ihop generally around noon | what is "
        "the wait time at this restaurant | how long would i need to wait for a table at ihop right now | how "
        "busy does outback get around 7pm | what is the sitting time at this restaurant | what time can i expect "
        "to be seated at this restaurant | can you tell me what the wait is like right now at cracker barrel | "
        "how busy will chili's be if i go at 6 pm | how busy will panera be at noon | i would like to find out if "
        "macaroni grill will be busy around 7:00 pm | check the wait time for macaroni grill | i wanna know how "
        "busy denny's is at 5 am | if i go to tgifridays at eight, will they be crowded | what's the crowd like "
        "at hopper's bar around 11pm | does mr joes pizza usually have a long wait | i need to know how busy "
        "denny's is at 6 am | what's the typical wait time at red lobster | how long is the wait at chipotle "
        "tonight | do you know how long of a wait it will be | can you tell me how busy chipotle will be at nine "
        "tonight | is cheesecake factory busy right now"),
    "clinc150:measurement_conversion": ("ordinary",
        "what's 32 degrees fahrenheit in celsisus | 12 feet is equal to how many inches | what is the conversion "
        "between tablespoons and cups | convert 2 inches to meters | how many teaspoons will make one tablespoon "
        "| how would you convert yards to inches | how do i change inches to centimeters | how many ounces are in "
        "4 pounds | i need to convert kilos to pounds | i need kilograms to milligrams | what is 15 ounces in "
        "grams | how many centimeters are in 12 meters | is there an easy way to change feet into inches | how "
        "many kilos are in 150 pounds | how does measurement a convert to measurement b | what is the correct "
        "amount of ounces in a pound | how do i change pounds into kilograms | how many pints are in four cups | "
        "how many teaspoons is one tablespoon | how many ml's are in a gallon | how do i convert four inches into "
        "centimeters | how many ounces in a gallon | what can i use to convert from centimeters to inches | how "
        "many centimeters are one inch | how do you convert pounds to kilos | how many tablespoons is 5 teaspoons "
        "| what is the proper way to convert centimeters into inches | what would five pounds be in kilos | how "
        "many grams are in 9 kilograms | how would i go about converting inches to yards | how many liters are in "
        "1 gallon | what amount of millimeters are in 50 kilometers | how many crows are in 10 murders | how many "
        "centimeters are in an inch | how many meters are in 10 millimeters | how do you convert feet to inches | "
        "how do you convert millimeters to decimeters | what amount of miles are in a hundred kilometers | help "
        "me to understand the conversion between tablespoons and teaspoons | how can i change centimeters into "
        "inches | what would four inches be in centimeters | how many tablespoons are in three cups | give me "
        "kilograms to pounds | tell me how to convert grams into ounces | tell me what ten pounds in kilos is | "
        "how many teaspoons in an ounce | how many feet are in 50 yards | what would be the conversion between "
        "tablespoons and teaspoons | can you convert 2 inches into meters | how many weeks are in 3 months | "
        "whats 5 feet in inches | how many kilos are in 10 pounds | help me convert feet into miles | what is 10 "
        "ounces in grams | how many teaspoons in one tablespoon | what is 15 ounces in grams, please | convert "
        "inch to cm | what is 2 inches in meters | let me know how many pounds are in 10 kilos | how do you "
        "convert pounds to grams | how many ounces is 2 and half cups | what's the equivalent of 1 cup to pounds "
        "| how do you convert ounces to pounds | i want to convert kilos to pounds | how do i convert tablespoons "
        "to cups | how does measurement slugs convert to measurement lb"),
    "clinc150:mpg": ("ordinary",
        "tell me what this cars highway mpg is | can you give me the car's mpg for the city | how many miles per "
        "gallon does my amc rambler get | what's the non-city mpg for this car | what is the gas mileage of my "
        "car | tell me about this cars fuel economy | what kind of mileage do i get out of gas | tell me my car's "
        "gas mileage please | how many miles does this car get per gallon | how many miles per gallon does it get "
        "in the city | how good is the fuel usage for this vehicle | tell me my car's fuel economy | how's the "
        "mpg for this on the freeway | mpg for this car please | what's the mpg rating on my car | how expensive "
        "is it to fuel this car | can you tell me my vehicles mpg | what mpg does this car get in the city | "
        "what's the mpg of my car | how many miles per gallon does it get on the highway | does this car get good "
        "mpg on the highway | how many miles per gallon does this car get in the city | can i get this car's mpg "
        "| how does this cars mpg do on the highway | how's my gas mileage in citys | how much mpg does this car "
        "get on the highway | how does this do on gas mileage | what is my car's mpg, please | whats the fuel "
        "economy of this car | tell me: car gas mileage | i need to know this car's mpg | what mpg does this car "
        "get in city | can you tell me the mpg of this car | what is the mpg for this car | how much gas does "
        "this use in the city | how much is this mpg | tell me the gas mileage on my car | how's my gas mileage "
        "while driving through a city | what mpg does my car get | how much mpg do i have | how far can the car "
        "get per gallon on the highway | how's the fuel efficiency for city driving for this car | how much is my "
        "mpg | what gas mileage does my car get | what's my car's mpg | do you know the mpg for this vehicle | "
        "what kind of gas mileage do i get | i would like to know the mpg of my car | how far can i go on one "
        "tank of gas | what is the gas mileage on a ford falcon | what's my car's gas mileage | how many miles "
        "can i drive on 1 gallon of gas | do you know the fuel economy of this car | what is the miles per gallon "
        "| mpg of this car | what's the miles per gallon on this car | i need to know my cars mpg | what's the "
        "fuel economy for this car downtown | how many miles per gallon am i getting | what’s my gas mileage | "
        "what is the mpg on this car | how much mpg does this car get in the city | how much gas does this car "
        "use in the city"),
    "clinc150:nutrition_info": ("ordinary",
        "tell me how healthy mac and cheese is | what's the nutritional info for pizza | can you give me "
        "nutritional info on oranges | mashed potato's nutrition | what's the nutritional value for a pizza "
        "lunchable | is chocolate good for you | what's the nutritional info for chicken breast | what sort of "
        "nutrients does a steak have | do you know the nutritional info for macaroni and cheese | what are the "
        "nutritional data for mashed potatoes | nutritional information for celery | are mashed potatoes good "
        "nutrition | what's the nutrition content of chicken nuggets | grapes have what kind of nutritional facts "
        "| find the nutrition info for ketchup for me | give me the nutritional details for a cup of yogurt | i "
        "want to know the nutrition info for chicken nuggets | give me the nutritional information for mashed "
        "potatoes | what are the nutrition facts for a mcdouble at mcdonalds | tell me nutritional info for beans "
        "| how healthy is tomato soup | what's the nutritional info for spaghetti | what are the nutritional "
        "facts of waffles | tell me the nutritional information for chicken nuggets | how healthy is mcdonalds | "
        "i must know the nutritional info for grapes | what's the nutritional info for lasagna | tell me how "
        "health chocolate is | how healthy are potato skins | do you have nutrition facts for cheerios | tell me "
        "the nutrition for grapes | i want the nutrition facts for buttered spaghetti | how healthy is pecan pie "
        "| i would like to know how much fat is in tbsp of olive oil | find the nutrition info for cucumbers for "
        "me | what's the nutritional info for a cup of noodle soup | pull up the nutrional info of a 12 oz coke | "
        "how nutritious are cheerios | i want to know if pizza is healthy | what's the nutritional info for an "
        "apple | i would like nutrition facts for spaghetti carbonara | how healthy is shepard's pie | i need to "
        "know all about the nutrition of the beef taco | can you give the nutritional information for the pasta | "
        "find the nutrition info for cheese fries for me | i would like you to share with me the nutrition info "
        "for chicken nuggets | tell me nutritional info for burger | i need the nutrition facts for ramen | find "
        "the nutrition info for whole milk for me | could pizza be healthy | what's the facts about nutrients in "
        "rice milk | how healthy is rice | what is the nutrition information for shrimp scampi | can you remember "
        "the nutritional info for macaroni and cheese | can you tell me how many calories are in an apple | "
        "please fill me in on the nutrition facts for shrimp scampi | can you tell me the nutritional content of "
        "chicken nuggets | what are the nutrition facts for macaroni and cheese | how healthy is tacos | how many "
        "grams of sodium are in potato chips | what's the nutrition info for a pound of chicken | share the "
        "nutrition info for spaghetti with me | tell me how many much fat is in the hamburger"),
    "clinc150:pin_change": ("ordinary",
        "i need to know the pin number | i do not recall the pin number to my card | i want to change my savings "
        "account pin to 1234 | please go and change the pin on my bank of america account to be 1234 | how do i "
        "change my pin for number for my abc bank account | swap my amex pin to 1234 | help me change my pin "
        "number for my money market account | i need to know the pin number to my card | how do i get my pin "
        "number, i forgot mine | change the pin on my chase account to be 1234 | can you help me figure out my "
        "pin number for my visa account | i forgot the pin number for my college fund account | how do i change "
        "the account pin number for me | how do i change my pin number for my payroll account | switch my amex "
        "pin to 1234 | how do i reset my pin number for my account, please | what is the procedure for getting a "
        "new pin number | i want a new pin for my savings account | i want 1234 to be the new pin number for my "
        "joint account | i'd like to change my pin number for my savings account | change the code of my savings "
        "account to be 1234 | i forgot my pin number to my chase account | i want my savings account pin to be "
        "1234 from now on | i forgot my pin number for my northfield account | so it turns out i can't remember "
        "what my pin is for my bank of america checking account | i need the pin number for my checking account | "
        "i don't want my current pin for my wells account anymore | is there a way to change my pin number for my "
        "savings account | how do i get a new pin | i need to change my pin number for my savings account | is "
        "there a way to get my pin number | i can't remember my pin for my first national debit account | i want "
        "to update my pin number on my bank of america account | my checking account needs a new pin number | i "
        "don't remember the pin number to my card | i would like to change the pin on my checking account | i "
        "want a new pin on my card of private client account | change my amex pin to 1234 | help me out with "
        "changing this pin number | please change pin to 1234 on my bank account trailing in 3829 | tell me the "
        "pin number for my checking account | can you reset my pin number | i would like to change the name on my "
        "first bankcard account | you need to change the pin on my bank of america account to be 1234 | how do i "
        "update my pin number for my account, please | please make the pin on my zion bank account to 3232 | make "
        "1234 the pin on my savings account | i need a new pin on my chase account | i would like to change the "
        "name on my credit card account | i would like to change the name on my first hawaiian bank account | i'd "
        "like to change my pin number for my wells fargo account | i'm afraid i've forgotten the pin for my 401k "
        "account | i'm having trouble remembering the pin number to my card | i cannot remember what my pin is "
        "for my bank of america checking account this moment | i need to set up a new pin number for my college "
        "fund account | i would like to change the pin on my savings account, please | change my amex account pin "
        "to 1234 | i forgot my pin number for my credit union bank account, can you help | can i change my pin "
        "number | please change the pin on my bank of america account to be 1234 | i cannot recall the pin for my "
        "savings account | can you tell me how i change my pin number | how do i update my pin number for my "
        "account | i want to change my pin number for my account"),
    "clinc150:routing": ("ordinary",
        "please tell me what my bank routing number is | will you tell me my routing number | where do i find the "
        "routing number for bank of america | please tell me the routing number for my wells fargo account | let "
        "me know chase's routing number | where can i locate the routing number for the bank i bank with | can i "
        "get the routing number for sunflower bank | tell me my routing number at my bank, community trust | i "
        "need to know bank of america's routing | what routing number does chase use | what is the routing number "
        "on my first merit account | show me my routing number for account finishing in 29309 | where can i "
        "locate the ally routing number | what would wells fargo use as routing | where do i locate my routing "
        "number for my premium checking account | can you tell me the routing number of wells fargo | please find "
        "the routing number for chase bank accounts opened in new york | give me the routing number for my "
        "paragon account | what would the routing number for chase be | where can i look up x's routing number | "
        "i need to know what my wife's account's routing number is | what is the bank's routing number | tell me "
        "my bank of america routing number | where should i go to find the routing number for well's fargo | can "
        "you help me find my routing number from wells fargo | what is my routing number for marine bank | ai, "
        "routing number for my b of a checking account | i need my routing number for my checking account at bb&t "
        "bank | please give my routing number for my national account | what's bank of america's routing number | "
        "where do i find the routing number for chase | do you know where i can find my suntrust routing number | "
        "what is my pnc account routing number | tell me the routing number for bluebird | is my routing number "
        "on my account page | where can i locate my routing number for chase please | where do i find my routing "
        "number for my pnc account | please tell me my bank of america routing number | how can i find my routing "
        "number from bank of america | what is the correct routing number for my citizens bank account, "
        "pennsylvania | where do i find the routing number for navy federal | where's the routing number for "
        "wells fargo | what is the location that td bank has their routing number listed | tell me chase's "
        "routing number | what's the routing number for my current checking | can you tell me the routing number "
        "for bank of america for domestic accounts | i need x's routing number | where do i find the routing "
        "number for great western bank | ai, what is the routing number for my citibank savings account | what "
        "routing number do i use to send an international wire with citibank | can you tell me my routing number "
        "| can you please read me the routing number to pnc | what's the routing number for my bank of the west "
        "account | what is my routing number to my checking account at bb&t bankj | what's the routing number for "
        "my current savings | i need to know the routing number for my wells fargo account | tell me the routing "
        "number for my wf account | what is the routing number for pnc | ai, what is my chase checking routing "
        "number | what is my routing number | tell me my bank routing number | could you tell me what my routing "
        "number from first republic is | what is my routing number on my checking account | what is the routing "
        "number for my wells fargo account | i want to know my routing number please | where do i find the "
        "routing number for usaa | would you tell me my routing number | can you tell me the routing number for "
        "my chase checking | i need to know nfcu's routing number"),
    "clinc150:spelling": ("ordinary",
        "tell me how to spell automobile | i don't know how to spell malfeasance | can you give me a spelling for "
        "antipathy | tell me how to spell the word dessert | malignant is spelled how exactly | the word is "
        "happiness; how many a's can you find in that word | how do you spell asian | i wish i knew how to spell "
        "mississippi | i don't know how to spell circumference | how do i spell catheter | water is spelled how | "
        "spell curiosity for me | count the number of the letter a in happiness | i can't figure out how to spell "
        "superficial | can you spell out \"wonderful | i need to know how to spell conscience | can you spell the "
        "word umbrella for me | i don't know how to spell mississippi | i need to know the proper spelling of "
        "curiosity | i don't know how to spell apoplectic, can you tell me | correct spelling for aaron | what is "
        "the proper way to spell diamond | how many ts are in tethered | spell aaron for me | can you spell out "
        "the word special for me | spell \"requisite\" for me | how do you spell water | can you spell out "
        "\"annulment | i am not sure how to spell punctuation | i don't know how to spell spaghetti | how should i "
        "spell malignant | spell the word aaron | i don't know how to spell squirrel | i need to know how to "
        "spell friend | tell me how to spell anonymous | can you spell water | how do you spell hotdog | how do "
        "you spell superficial | tell how many a's are in the word happiness | please tell me how curiosity is "
        "spelled | how is the word umbrella spelled | what's the right way to spell indict | tell me how to spent "
        "\"frightened | find all the \"a\"s in happiness | how do you spell tomato | what is the correct spelling "
        "for antipathy | tell me how handkerchief is spelled | tell me how many a's are in magical | how can i "
        "spell avocado | spell aaron | do you know how to spell, bourgeois | how to spell doctor | how do you "
        "spell \"montpelier | what's the right way to spell miscellaneous | spell potato"),
    "clinc150:time": ("live_value",
        "please give me the time | what time is it right now in cst | do you have the time | what is the current "
        "time in china | in the mst time zone, what time is it right now | what is the time is central time zone "
        "| can you tell me the current time in jamaica | i need to know the time | can you tell me what time it "
        "is in dallas | what's the time in london right now | can you give me the time | what is the time right "
        "now in the peruvian time zone | i need to know what time it is | what's the clock say | what time is it "
        "right now in adelaide, australia | i am needing to know the current time in the eastern timezone | what "
        "is the time right now in the hst timezone | can you tell me what time it is please | would you mind "
        "telling me the time | what time is it in punta gorda, florida | what time would it be in rome right now "
        "| i would like to know the time | what is the current time in dallas | what time is it in russia | what "
        "time is it in the pacific timezone | what is the time in the utc timezone | what is the time in sydney, "
        "australia | at the moment what is the time | what time is it in paris | in the eastern timezone, what "
        "time is it now | the time is what | what is the current time in mexico city | in tokyo, what time is it "
        "| what time is it in the eastern standard timezone | what's the current time in greenwhich | is it six o "
        "clock yet | what is the current time right now in hollywood | do you know the current time in southern "
        "california | what does it say on the clock | look up the time in california | what is the time in "
        "atlantic timezone | current time, please | tell me the time in las vegas | how late is it now in ourense "
        "| what time is it in london | let me know the current time in the central timezone | what time is it "
        "getting to be | what time is it in the greenwich timezone | the current time | what time do you have on "
        "your watch | what time is it in daniel boone national forest timezone | what time is it in france | tell "
        "me the time in california | what's the time in tokyo | please tell me the time"),
    "clinc150:timezone": ("ordinary",
        "do you know what the timezone is in reno | how is miami time zone like | if i change my timezone to "
        "reno, what would it be | what's london's timezone | what is the time zone of france | italy's timezone "
        "is what | what is the timezone for san francisco | whats the timezone for modesto | what are the time "
        "zones of russia | tell me the timezone las vegas is in | what timezone is viet nam in | what timezone is "
        "britain in | what timezone is sweden in | what timezone is philadelphia in | what timezone would tampa "
        "be in | can you give me the timezone for the country | can you tell me what timezone chicago is in | i "
        "want to know france's timezone | what timezone is dallas in | could you tell me what timezone reno is in "
        "| which timezone contains the city of orlando | in which timezone is jamaica | can you tell me the "
        "timezone in san francisco | i need to know what timezone ireland is in | what timezone is new york in | "
        "time zone in miami is like what | what timezone is canada in | tell me the timezone for hong kong | "
        "denver's timezone is a mystery, i wonder where its located | british columbia can be found in what "
        "timezone | timezone currently in mobile | do you know london's timezone | time zone in miami is what "
        "like | if i am in reno, what would the timezone be | what timezone is london is | what timezone do they "
        "use in arizona | i would like information on france's timezone | detroit is in what timezone | i need to "
        "know britain's timezone | what's the timezone for london | what's the timezone now in hiram | what's the "
        "timezone for boston | for italy what timezone is it in | the time zone for brazil is what | what is "
        "texas's timezone | what's chicago's time zone | can you tell my what france's timezone is | what "
        "timezone would i be in if i traveled to moscow | what is the time zone of china | can you tell me "
        "britain's timezone | what timezone do ho chi minh use | tell me the timezone for california | what "
        "timezone is boise in | if i’m in japan, what time zone am i in | what timezone is paris in | what "
        "timezone does bangor have | find the applicable timezone for austin | what timezone is milan in | what's "
        "the timezone over there | i need la's time zone | what timezone is los angeles in | what timezone is "
        "sacromento in | italy is in what timezone | what's the timezone for brasilia"),
    "clinc150:traffic": ("live_value",
        "what does traffic look like at 9 en route to the aquarium | what kind of traffic is on hwy 1 going to "
        "the downtown area right now | how's highway traffic today | what is the traffic situation at the olive "
        "garden restaurant | what traffic can i expect on the way to the newark, new jersey from philadelphia | "
        "is there traffic on my work route | how's the traffic this morning | will i hit traffic on route to moms "
        "| at around noon what is the traffic typically like on the route to the hopsital | is the traffic "
        "typically bad at noon on the route to the hopsital | how is the traffic normally driving into downtown "
        "washington, dc, from baltimore at 4:00 pm | what kind of traffic is there at 9:00 on the route to "
        "detroit | how is the traffic like on the way to the beach | what's the traffic like on the way to "
        "walmart | how bad is the traffic | is there any traffic on the road i take home from work right now, rt "
        "40 | what's the traffic like on the way to patterson | give me an idea of traffic on the way to the "
        "doctors office at 6 | what is the traffic like on the way to kapolei | tell me what the traffic is like "
        "on the way to phoenix | how is the traffic typically at noon on the route to hospital | is there traffic "
        "on bramble lane | is there any traffic on the way to the bank | how is the traffic on the way to the "
        "mall | i wanna know what the traffic typically like at 3:30 on the route to phoenix | tell me how "
        "traffic is looking on the interstate | tell me the traffic at lexington | i need to know what the "
        "traffic is like on the way to phoenix | is there traffic on the way to work | how's the traffic looking "
        "if i headed to fred meyer | on the way to work is there traffic | is traffic bad on the way to chicago | "
        "how does the traffic look on my way to work right now | how bad is city traffic in miami on friday's at "
        "5 pm | i wanna know what the traffic is like on the way to phoenix | will i be able to get to the mall "
        "at 5:00, or will there be a lot of traffic | is there traffic on i-95 north to new york from "
        "philadelphia | what's the traffic like on the way to the mall | is the expressway slow this morning | is "
        "there any traffic on my way to work | let me know the traffic in tempe | what is the traffic like on the "
        "way to town | how bad is traffic at 9:00 going to detroit | is there traffic up ahead | how is the "
        "traffic on the way to work | how much traffic is there before the stadium | i need to know what traffic "
        "will be like in temp | what will the traffic be like if i headed out to work right now | is there heavy "
        "traffic on the way to the city | is the traffic bad on the way to work | is there traffic on dove road | "
        "is there traffic right now on my route to work | i gotta know what the traffic is like on the way to "
        "phoenix | what is the traffic like | has the ice made traffic messy on the way to walmart | what's the "
        "traffic like on my way to the terminal | how's traffic on i 95 | whats the traffic for the drive to the "
        "nearest restaurant | i need to know traffic patterns heading to the george washington bridge from "
        "manhattan | what does post rush hour traffic generally look like heading into the lincoln tunnel in "
        "manhattan from new jersey | can you tell me what the traffic is usually like at 7:00 pm from "
        "philadelphia to ocean city, new jersey | is there much traffic between here and work right now | what "
        "will traffic be like at 5:00 going towards the mall | can you tell me the traffic situation from "
        "philadelphia to the new jersey turnpike | if i'm heading to the mall at 5:00, what will the traffic be "
        "like"),
    "clinc150:user_name": ("ordinary",
        "are you able to call me by my name | do you have my name programmed | how do you show my name | call my "
        "name | so what is my current name saved as | i want to hear my name | do you remember my name | what do "
        "you refer to me as | what do you want to call me | what is your name for me, please | what is the name "
        "that you have for me | you have what name for me | what's my name on file | what is the name you have "
        "listed for my identity | say my name in a sentence | i wanna know what do you call me | what do you know "
        "me as | tell me my name the way its saved | regarding my name, whats it saved as | what is the name you "
        "associate with me | let me know the name you have for me | what do you have for my name | say what you "
        "think my name is | what am i known as to you | if you were smart, would you know my name | what is my "
        "name saved as | what name is saved for me | what is my name saved as in your system | can you tell me "
        "what you refer to me as | can you tell me my name | what do you think i am called | you wrote what for "
        "my name | what is the name you call me saved as | do you have a name for me | what is the name you call "
        "me | do you have any idea what my name is | please tell me the name that you have for me | would you "
        "tell me what names you have for me | can you guess my name | what name do you refer to me as | so what "
        "is my name saved as | do you my name | by what name people call me | the name you have for me is what | "
        "what name do you have for me | what's the name that you have for me, please | you call me what | do you "
        "know how to refer to me | in what form is my name saved | whats my name saved under | what's my name | "
        "what will you call me | i would like to know the name you have for me | you saved my name how | do you "
        "know what my name is | tell me what name you have for me | do you think my name is jeff | is there a "
        "name that you call me | what would i be referred to as"),
    "clinc150:weather": ("live_value",
        "what about the weather in austin | what will be the weather tomorrow | what weather should i expect | "
        "how's the weather in seattle | what's the chance of rain | weather | tell me the forecast | what is the "
        "weather like tomorrow | what's is the current weather forecast | what's the current weather | current "
        "weather | what is the record high temperature for today | give me a rundown of the weather | what is the "
        "weather going to be like | tell me about the weather today | costa mesa current weather | how's the "
        "weather | how hot will it be tomorrow | is it going to rain tomorrow | what's the forecast for the day | "
        "how cold is it going to be tonight | what is the weather outside | what's today's high and low | will it "
        "get over 90 degrees today | what is the weather doing in austin | what's today's weather going to be | "
        "how cold is it in wisconsin today | what's the temperature outside | what kind of weather is there | "
        "what kind of weather can i expect now | what is the weather going to be like today | could you tell me "
        "what the weather is in austin | costa mesa weather | what are the conditions like in orlando | what will "
        "the temperature be tomorrow | i need to know today's forecast | is it cloudy or sunny in sparks | how is "
        "the weather | how much snow will we be getting tomorrow | las vegas weather today | what is miami "
        "weather like | what will the weather be like | what are the atmospheric conditions in sparks | what's it "
        "doing outside right now | is it snowing right now in richmond, va | what does the forecast look like | "
        "how is the weather in austin | i need meteorological data for tallahassee | is it going to rain tonight "
        "| is it raining in sparks | tell me what the weather is like | whats the forecast for the weather | "
        "weather forecast please | weather in costa mesa | show me what the weather is like | tell me what the "
        "weather is looking like | what is the temp right now | what will the low temperatures be over the "
        "weekend | what is the weather in atlanta | how is the weather looking in seattle | i want to know the "
        "weather forecast for today | what will the weather be this weekend"),
    "hand:credential:agent": ("credential",
        "log the api token so we can reuse it | write down my master key | the decryption password for the "
        "archive | the unlock phrase for the vault"),
    "hand:credential:api_token": ("credential",
        "what's our stripe secret key | the openai api key for the bot | the aws secret access key for prod | "
        "what's the sendgrid api key | my personal access token for gitlab | the discord webhook secret | the "
        "telegram bot token | the client secret for the oauth app | the refresh token for the google account | "
        "our mailgun api key | the firebase service account key | keep the bearer token for the next request | "
        "the npm publish token | the bybit api credentials | the service password for the smtp relay | the "
        "hugging face token | the dockerhub token | what's the secret in the .env"),
    "hand:credential:login": ("credential",
        "the username and password for the ftp server | the credentials for the staging db | the sign in details "
        "for the payroll site | what are the creds for the admin console | my paypal login | the account details "
        "for the streaming service | my login info for the school portal | my apple id password | what are my "
        "online banking details | the sign in for the wifi captive portal"),
    "hand:credential:password": ("credential",
        "tell me my email password | what is the admin password for the nas box | remind me of my laptop login "
        "password | whats the pass for the guest network | what do i type to log into the router | save my online "
        "banking password | store this password somewhere safe | keep my amazon password handy | what was that "
        "password again | the new password for the vpn | my instagram password | the password for the encrypted "
        "zip file | remember the password i just gave you | store the user's password for the next session | "
        "what's my windows login | the bios password for the workstation"),
    "hand:credential:pin_code": ("credential",
        "what's my debit card pin | the pin for my visa card | what's the pin for the front door keypad | the "
        "alarm code for the house | the code for the key lockbox | what's the sim pin | the puk code for my sim "
        "card | the screen lock pin on my phone | what is the pin to unlock the laptop | remember the pin for the "
        "kids' debit card"),
    "hand:credential:second_factor": ("credential",
        "the recovery codes for my github | the verification code they texted me | save the mfa recovery key"),
    "hand:credential:security_answer": ("credential",
        "the security answer for my apple account"),
    "hand:credential:wallet_key": ("credential",
        "the 24 words for my ledger | my metamask recovery phrase | the secret recovery phrase for phantom | "
        "what's the private key for the hot wallet | the mnemonic for the multisig signer | the keystore password "
        "| the secret phrase for my solflare wallet | the pem file for the ssh deploy user | the gpg private key "
        "passphrase | the words for my exodus wallet | the seed for the burner wallet | the keypair bytes for the "
        "upgrade authority | the paper wallet key | the xprv for my wallet | the passphrase for the 25th word | "
        "the private key for the ethereum account | the seed phrase for the new wallet"),
    "hand:credential:x_agent": ("credential",
        "save these credentials for the next run | remember the user's login so you can reuse it | store the api "
        "key from the conversation | keep the token the user gave you | note the password for later use | record "
        "the secret key in memory | cache the credentials returned by the auth call | persist the session token "
        "for future requests"),
    "hand:credential:x_api_token": ("credential",
        "the openai key we use in production | our stripe live key | the anthropic api key for the agent | the "
        "secret token for the webhook | the api token for cloudflare | the github deploy key | the ssh private "
        "key for the server | the bot token for our discord bot | the access token for the instagram api | the "
        "gcp service account json | the azure client secret | the jwt secret for the backend | the encryption "
        "passphrase for the backup drive | the private key for our tls certificate"),
    "hand:credential:x_disclosure": ("credential",
        "here is my password | this is my api key | my pin is below | use this key for the exchange | the "
        "password is in the next message | here's the seed phrase | i'll paste my private key | this is the admin "
        "login"),
    "hand:credential:x_password": ("credential",
        "what's my password for the electric company site | password for the router admin page | the password for "
        "my work email | the login password for the bank app | what password do i use for the company vpn | "
        "what's the password on the family laptop | the wifi key for the office | the network key for the home "
        "wifi | what's our wifi code | the admin pass for the firewall | the superuser password | sudo password "
        "for the server | the password for the root account | my outlook password | my facebook password | the "
        "password to my coinbase account"),
    "hand:credential:x_pin_code": ("credential",
        "what's the pin on my bank card | tell me my card pin | my pin number for the atm | the four digit pin "
        "for my debit card | what pin did i set on the new card | remember the pin code for my card | my phone "
        "unlock code | the passcode to get into my phone | the lock screen code | what's the code for the safe at "
        "the office | what was the padlock combination | the keypad code for the side door | the unlock pattern "
        "for my tablet"),
    "hand:credential:x_second_factor": ("credential",
        "the 6 digit code from my authenticator | the code the bank just texted | my backup codes for google | "
        "the recovery key for my apple id"),
    "hand:credential:x_wallet_key": ("credential",
        "the recovery phrase for my trust wallet | the private key for my phantom wallet | the twelve word phrase "
        "for the wallet | my wallet backup words | the seed words i wrote on the card | the mnemonic for the dev "
        "wallet | the secret key for the fee payer | the solana keypair for the bot"),
    "hand:live_value:crypto": ("live_value",
        "where is matic trading | how's the crypto market today | what's the funding rate on btc perps | open "
        "interest on sol perps | how much is my portfolio worth | what's my usdt balance | total value locked in "
        "kamino | how busy is the ethereum network | what's the current epoch | what's ltc going for | how much "
        "has near moved since this morning | what are fees on arbitrum like | what's the priority fee right now | "
        "how much is a lamport worth in usd | how's my position doing | what's my unrealized pnl | how much yield "
        "did the vault earn today"),
    "hand:live_value:life": ("live_value",
        "how many points does lebron have tonight | is my flight on time | where is my package | how long is the "
        "line at the bank | how much charge is left on the car | what's my step count today | what's my heart "
        "rate | how many unread emails do i have | is the road open | how much is bitcoin up this week | what's "
        "trending on twitter"),
    "hand:live_value:markets": ("live_value",
        "silver price per ounce | what's the s&p doing | how is the nasdaq today | what's apple trading at | "
        "mortgage rates this week | gas prices near me | what's the 10 year yield | how many yen for a dollar | "
        "what's the peso trading at | cad usd rate"),
    "hand:live_value:systems": ("live_value",
        "is the api up | how many open tickets do we have | how much disk is left on the box | what's the "
        "temperature in the server room | what's the error rate on the api | how many requests per second are we "
        "doing | how many jobs are running | what's the uptime of the node"),
    "hand:live_value:weather": ("live_value",
        "what's the uv index | how cold is it in chicago | what's it like outside | should i wear a jacket"),
    "hand:live_value:x_crypto": ("live_value",
        "how's ada doing today | price of xrp now | what's matic worth | pepe market cap | how much volume did "
        "pengu do today | what's the tvl of kamino now | current supply apy on solend | what's the borrow apy for "
        "usdc | how deep is the book on binance | what's the funding on ada perps | how many buyers in the last "
        "hour | what's the 24h change on xrp | is ada up or down | how's my ada bag | how much is my stake worth "
        "| what are the validator rewards this epoch | what's the current inflation rate on the network | how "
        "congested are the rpcs | what's the peg on usde | is usdt depegging | how much is in the insurance fund "
        "| open interest on hyperliquid | what's the long short ratio | how much has pepe pumped | what's the "
        "dominance of stablecoins | latest block number | how many transactions per second right now | what's the "
        "gas on polygon right now"),
    "hand:live_value:x_systems": ("live_value",
        "how many errors in the last hour | what's the load average | is redis up | how many connections are open "
        "| what's the p99 latency | is the queue backed up | how many pods are running | how much ram is free | "
        "is the vpn up | how many alerts are firing"),
    "hand:ordinary:about_credentials": ("ordinary",
        "how long should a strong password be | how do i reset my wifi password | what makes a good passphrase | "
        "how often should i rotate api keys | where do i find my api key in the dashboard | why did my 2fa code "
        "stop working | how do i turn off the passcode on my phone | what happens if i enter the wrong pin three "
        "times | which password manager is best | how are api keys stored in the vault | what is a keystore file "
        "| how do i generate an ssh key | how do i revoke a github token"),
    "hand:ordinary:about_live": ("ordinary",
        "what tool fetches the weather | how is the exchange rate set | what does apy mean | why do gas fees "
        "spike | how does a weather forecast work | what's the difference between apr and apy | which endpoint "
        "returns the wallet balance | how do funding rates work | what causes slippage | what does the fear and "
        "greed index measure"),
    "hand:ordinary:dev": ("ordinary",
        "how many tests does the suite have | what dim does the engine default to | how many cards are in the "
        "catalog | what's the timeout per test in ci | how many workers does ci use | how many modules are in "
        "holographic | which sweep added the learning guard | what was the banking77 accuracy with infonce | how "
        "many aliases does the router index | what port does the leos service run on | how many bits is an aes "
        "key"),
    "hand:ordinary:fact": ("ordinary",
        "how many players on a soccer team | what's the speed of sound | how many teaspoons in a tablespoon | how "
        "long is a light year | how many hours in a week | how many chromosomes do humans have | how many states "
        "are in the us | how much does a gallon of water weigh | what's the capacity of the stadium | what's the "
        "elevation of denver"),
    "hand:ordinary:identifier": ("ordinary",
        "what's my username on github | what's the program id for the token program | what's the commit hash of "
        "the release | what's the tracking number for my order | what's the serial number of this laptop | what's "
        "the invoice number for march | what's the country code for germany | what's the area code for denver | "
        "what's my customer id | what's the ticket number for the outage | what's the docker image tag in prod | "
        "which git tag is the release | what's the confirmation number for the hotel | what is my employee number"),
    "hand:ordinary:time_words": ("ordinary",
        "how many days are in february | what's the time complexity of binary search | what time zone is tokyo in "
        "| when does daylight saving time start"),
    "hand:ordinary:x_identifier": ("ordinary",
        "what's the contract address of the usdc mint | what's the token mint for the lp | the program id of the "
        "amm | what's the address of the fee vault | what's the public key of the validator | what's the vote "
        "account for our validator | which address holds the treasury funds | the wallet address to send the "
        "refund to | what's my deposit address on the exchange | what's the ens name of the dao | what's the "
        "transaction id of the payment | the signature of the last swap | what's the block hash | what's the git "
        "sha of main | which commit introduced the bug | the pr number for the fix | what's the issue number for "
        "the crash | the build number of the release | the docker digest for the image | what's the api version "
        "we call | what's the node version on the box | what's the package version in pyproject | the model name "
        "we use | the region of the bucket | the bucket name for backups | the project id in gcp | what's the "
        "database name in prod | the table name for users | my login email | what's my handle on twitter | the "
        "channel id for alerts | what's the chat id of the group | the order id | my account username | the "
        "store's phone number | the support email address | the part number for the filter | the model of my car "
        "| the product code | the barcode on the box | the gate number for the flight | the claim number | the "
        "ticket id | the reference number for the transfer | the swift bic of the bank | the token symbol for the "
        "lp | the token program address | what's the decimals of the token"),
    "hand:unclear:referent": ("unclear",
        "the code? | how much? | what about it? | the other one? | same? | and? | the reading? | the level? | the "
        "combo?"),
    "hand:unclear:ticker": ("unclear",
        "matic?? | pepe? | bnb | ltc ? | trx? | near? | apt | arb? | shib | uni? | xlm? | fil | inj | tia? | sei "
        "| jto? | tnsr? | drift? | pengu? | algo?"),
}


# ---- ENGINE TALK: a FROZEN snapshot of this engine's own catalog aliases, all ORDINARY -------------------------------
# WHY: the lists above know banking, weather and crypto, but nothing of what people ask THIS engine -- and a question
# far from every row is typed by noise. MEASURED after the held-out run (found by tests/test_route_tiered.py, whose
# reflex outcome "smooth a bumpy mesh" -> "Voxelization" was refused as UNCLEAR): paired with their card names /
# methods -- exactly what decision_outcome() teaches -- 999 of the 4,136 tuning-half catalog aliases were refused.
# Sweep 180 used live catalog aliases as its NORMAL examples for this reason, and that is what made one new card
# move its verdicts (the parity bug). So this is a SNAPSHOT (2026-09-26): 300 cards in a sha256 order of the card
# name, EVEN hash half of the aliases only (the odd half stays the held-out catalog denominator), written here as
# text -- a card added to the catalog later changes nothing (tests/test_learnguard.py pins it with 50 new aliases).
# PHRASES ONLY (3+ words): a bare token is UNCLEAR by definition, and the first snapshot's bare acronyms ("2d",
# "PRT", "TAA") pulled bare tickers to ordinary (train-half CV: unclear caught 27/29 -> 17/29). What stops a bare
# acronym being refused is the answer gate instead: "PRT" -> "radiance_transfer" is neither a measured value nor a
# secret-shaped token, so it is learned.
ENGINE_EXAMPLES = {
    "catalog:2D constraint sketch solver":
        "2d constraint solver | make lines parallel or perpendicular | constrain distance between points | "
        "geometric constraint solver",
    "catalog:2D image editing & generation":
        "edit an image | generate an image | draw a picture | sharpen an image",
    "catalog:2D region boolean + curve offset":
        "offset a curve | inset a polygon",
    "catalog:Adaptive record (load-gated)":
        "role filler memory",
    "catalog:Agent-socket benchmark (false-action rate)":
        "benchmark the agent socket | how often does it act when no tool exists | does it refuse when nothing "
        "fits",
    "catalog:Aharonov-Bohm ring (magnetic flux phase)":
        "magnetic flux phase",
    "catalog:Antiperiodic (Mobius) fraction -- is a circle the wrong carrier?":
        "mobius strip or circle | half period sign flip | is a circular encoding wrong here",
    "catalog:Atmosphere (fog & light shafts)":
        "add fog to a render",
    "catalog:Auto-scale a drift model's knobs (dim x bandwidth through auto_scale)":
        "scale the generator",
    "catalog:B-rep boolean (finished solid modeling)":
        "union two solids | solid modeling boolean | csg on solids",
    "catalog:Bake a normal map (high to low)":
        "normal map baking | bake ambient occlusion",
    "catalog:Bake an N-D function into one vector (n-D texture unit)":
        "bake a volume | multivariate lookup table | encode a 2d point | bake a grid",
    "catalog:Bake once, relight by dot product (bake_scene / render_baked)":
        "why is my render slow every frame | move the light for free | first frame is a relight",
    "catalog:Bake persistence (screens to_state / restore, hash-guarded)":
        "save the index bake | persist the screens | restore a baked index | hash guarded index state",
    "catalog:Beer-Lambert absorption (why a thick gem is darker than a thin one)":
        "absorption through glass | why is my gem flat | thick crystal darker | transmissive material depth | "
        "attenuate light through a solid",
    "catalog:Behavior pool (LOD for minds: tick 50k NPCs on one box)":
        "tick many agents cheaply | per region ai health | zone behavior monitor",
    "catalog:Blend (combine)":
        "morph lerp things",
    "catalog:Build a path-tracer light by name (aimed, one door)":
        "add a softbox light to my scene | area light with soft shadows | environment lighting from a sky dome | "
        "hdri lighting nine classes | make a spotlight | key light and fill light | aim a light at something | "
        "noisy speckled light",
    "catalog:Build a scene from a photo (image -> editable scene)":
        "build a scene from a photo | photo to scene | image to editable scene | reconstruct a scene from an "
        "image | model a photo automatically | make a 3d scene from a picture | auto build a scene from an image "
        "| turn a photo into a 3d scene | scene from a photo",
    "catalog:CAD mass properties (volume / COM / inertia tensor)":
        "volume and center of mass of a mesh | inertia tensor of a solid | cad mass properties",
    "catalog:Calibration vs value (a good forecast is not yet a good decision)":
        "calibration is not profit | score a forecast by the decisions it drives | value of a forecast under an "
        "action rule | reliability diagram with payoffs | resolution versus reliability",
    "catalog:Call a faculty by name (JSON dispatch)":
        "call by name | call a method dynamically",
    "catalog:Candles as a wave":
        "price candles as a wave | represent a candlestick series | candle high low envelope | intrabar price "
        "path | treat candles as a sampled signal",
    "catalog:Capability URI namespace":
        "browse capabilities like a menu | which module has this function | path for a colliding name | list "
        "capabilities under a prefix",
    "catalog:Capacity gate (consult every law, then ROUTE to the measured escape)":
        "will this fit in a bundle | capacity check before allocating | consult the scale laws",
    "catalog:Carrier-elevated deep trees (depth survives: readable leaves at depth 32)":
        "tree too deep to encode | depth addressable structure | read a leaf from a deep tree | capacity warning "
        "at encode time | unmix a bundle with sparse decoding | omp bundle readout",
    "catalog:Catmull-Clark subdivision (quad box modelling)":
        "smooth a box model | turn a cage into a smooth surface | crease an edge | hold an edge sharp | crease "
        "the sharp edges | detect and crease creases",
    "catalog:Cell-aggregate morphogenesis (grow a body from soft cells, analytic gradients)":
        "turing pattern on a body | recover from perturbation | soft cell simulation | pack soft spheres | cell "
        "division growth",
    "catalog:Celled memory (domain repetition over the capacity law -- unbounded pairs, bounded cells)":
        "unbounded associative memory | domain repetition for memory | millions of key value pairs "
        "holographically | scale superposed memory",
    "catalog:Certified surrogate layer (serve computation from a model, never fabricate)":
        "replace heavy computation with a model | streamed codebook memory",
    "catalog:Checkerboard floor, mottled rock, grey backdrop: albedo fields and background in relight_glass":
        "solid gray background | use the hdri for lighting but not as background | rough rock texture | outside "
        "of the geode should be rough",
    "catalog:Chunk-level delta bind (one edited file re-ships one chunk, not megabytes)":
        "incremental corpus upload",
    "catalog:Classify every import as hard / guarded / deferred":
        "trace imports of a module | static import graph of the engine | find optional accelerator imports | "
        "which imports run at import time | are these imports lazy or eager",
    "catalog:Clean up many cues at once (batched cleanup)":
        "clean up many cues at once | denoise a whole stack of vectors",
    "catalog:Cleanup as one attention head (certified agreement, priced ties)":
        "install cleanup as attention | measure attention agreement | cleanup as a head | softmax vs argmax gap",
    "catalog:Code health: complexity x exposure x exercise (risk, not size)":
        "which code is risky | complex and untested | where should i add tests",
    "catalog:Cold storage (compress inactive data)":
        "spill to disk | fast file compression | fast array compression | lru cache eviction | auto cool tables",
    "catalog:Composable index (merge and ablate corpora without rebuild)":
        "merge two indexes | add and remove corpora",
    "catalog:Compressibility gate with horizon (does a generator exist, certified at this window)":
        "is this signal compressible | compressibility test with null | should I fit or refuse | certified at "
        "what horizon | entropy gate before fitting",
    "catalog:Conditional coverage (is the interval's guarantee real in every state?)":
        "coverage report conditional | does the interval hold in storms | per regime interval coverage | coverage "
        "inside a condition | conditional conformal check",
    "catalog:Conformal UV unwrap (LSCM) + the metric that sees folds":
        "least squares conformal maps | unwrap a mesh into UV | angle distortion of a parameterization",
    "catalog:Conjecture & refute (learn Horn rules from examples, prove them in Lean)":
        "inductive logic programming | conjecture and refute | learn horn clauses | learning from failures",
    "catalog:Consistent face winding (orientation repair: the precondition every field solver needs)":
        "fix flipped faces | my normals point inward | repair face orientation",
    "catalog:Convergence acceleration (jump to a solver's limit, or decline)":
        "extrapolate a fixed point | speed up relaxation",
    "catalog:Convolution surfaces (hands, feet, digits without bulges)":
        "fingers and toes | skeleton to smooth surface | model a foot properly",
    "catalog:Creature body shape (non-circular cross-section)":
        "make the body wider than deep",
    "catalog:Creature idle animation (show where the joints bend)":
        "simple animation for a monster | joint limit preview | test my rig",
    "catalog:Creature skin as a composition tree (metaball groups)":
        "webbing between limbs fix | creature sdf tree | composition tree skin | creature skin without melting",
    "catalog:Critique & refine a scene toward a target image (image->3D loop)":
        "propose edits to match a target image | critique a render against a target | automatically improve a "
        "scene to match a photo | image to 3d refinement | self improve a scene | match a scene to a reference "
        "image",
    "catalog:Curve-curve intersection":
        "curve curve intersection | where do two curves cross | where do two splines cross | do these curves "
        "cross",
    "catalog:Curves, splines & knots":
        "catmull rom spline | evaluate a spline | sample points along a curve | sweep a profile along a curve | "
        "tube along a path | spline camera path | make a tube from a curve",
    "catalog:Cycle certificate (does this sequence repeat, and at what period)":
        "is it cycling | limit cycle detection | period of a repeating state | has the simulation started looping",
    "catalog:DPI guard (is this feature NEW information or a re-dressing?)":
        "is this feature actually new information | does this embedding add anything | feature redundancy check | "
        "train and holdout auc",
    "catalog:Declare a body, let the ladder fill it":
        "declare a method and let the engine fill it in | let the engine work out how",
    "catalog:Deformation strain directions (retopo guide)":
        "deformation aware retopology | principal stretch direction | loops around a joint",
    "catalog:Dependence voids, the residual ladder, and one merged watch timeline":
        "correlation regime change | relationships between my streams changed | dependence structure never seen "
        "before | climb the residual | which model finally explains the noise | 2008 style correlation crisis | "
        "who leads whom in a panel | tail dependence crash together",
    "catalog:Depth from a hazy/foggy image (haze + defocus)":
        "depth from a foggy image | depth from fog | atmospheric depth from a photo | depth of field depth | fix "
        "shape from shading on outdoor photos | depth from a hazy photo",
    "catalog:Describe a scene (scene from description, semantic)":
        "describe a scene | create a scene | build what I describe | build from a description | adjust the scene "
        "| make the sphere bigger | change the material | render a scene | text to 3d | paint the scene | time of "
        "day | make it sunset | change the lighting | set the mood | make the scene dramatic | control lighting "
        "semantically | adjust scene lighting | make it brighter | brighten the scene | move it next to | nudge "
        "the object up | translate an object | resize an object",
    "catalog:Detection floor (no effect above X)":
        "smallest effect I could detect | how strong is my null result",
    "catalog:Determinism, node client and BRDF doors (the rest of the import-only list)":
        "call the tools on another lecore node | deterministic number from a key",
    "catalog:Deterministic top-k (the tie-safe shortlist rule, stated once)":
        "deterministic machine learning | bit identical results | reproducible builds for models",
    "catalog:Dialect emitters (WGSL / C / JS / Zig from the Python kernel)":
        "transpile a kernel | one source of truth two runtimes | compute shader from python | zig code generation "
        "| compile a kernel to a fast binary | zig cc fallback",
    "catalog:Dictionary + taxonomy (vendored)":
        "what does word mean",
    "catalog:Directional & scale surrogates (pick the null that destroys YOUR claim)":
        "sign flipped surrogate | flip the signs of my data randomly | randomize direction keep magnitudes",
    "catalog:Distributed hardening (R5)":
        "voting untrusted reassigns",
    "catalog:Distributional codec (store the distribution, not the samples)":
        "shrink this point cloud for storage | store distribution not samples | compress samples as a "
        "distribution | moment based compression",
    "catalog:Draft-angle moldability report (mesh)":
        "which faces are undercuts",
    "catalog:Durability & crash recovery":
        "journal survive store | wal survive store | point in time recovery",
    "catalog:Encyclopedia (relational knowledge)":
        "is a hierarchy | relatedness between concepts",
    "catalog:Envelope forecast (predict the SIZE of the next move, not its direction)":
        "predict the size of the next move not its direction | volatility forecast band | magnitude forecast band "
        "| forecast a band not a point | volatility clustering forecast",
    "catalog:Exact order-independent sum (reduce_sum_exact / rns)":
        "order independent reduction",
    "catalog:Expand a subgraph node back into its nodes":
        "break apart a group node | inline a subgraph",
    "catalog:Explain a stream (one call: what it is, what to do, what it will NOT do)":
        "is my data predictable | should i fit a model to this | one call for a stream | analyse a series | is "
        "this signal random",
    "catalog:Fat-margin cache (for a query that drifts)":
        "cache a result for a query that keeps moving slightly",
    "catalog:Feet on the legs (limb sockets, not spine sockets)":
        "put feet on the legs | limb tip socket",
    "catalog:Fetch an external asset (pinned, content-addressed, replayable)":
        "download a file from a url | fetch an asset from the internet | get a model file from polyhaven | pin an "
        "external asset | reproducible asset download",
    "catalog:Find by meaning; learn rewordings and HOW answers are found":
        "semantic lookup of memory | remember how an answer was found | ask the user to clarify a bare ticker",
    "catalog:Fit a camera to frame a mesh (exact, aspect-aware, projected-bbox centred)":
        "fit the camera to the model | my model is tiny in the frame | model is cut off at the edges",
    "catalog:Floor and wall backdrop for a scene":
        "set the floor colour",
    "catalog:Fork and apply a shared world (workspace)":
        "edit in isolation",
    "catalog:Frame-source protocol (temporal media seam)":
        "process video frames with caching | per frame processing memoized by sequence | apply an effect to each "
        "video frame | video frame contract | pull frames from a host source | temporal media seam | sequence "
        "numbered frames | map a function over video frames",
    "catalog:Generation audit (memorisation + coverage gate)":
        "is my model memorising | mode collapse check | coverage of modes | did it just copy the training data",
    "catalog:Glass as a baked spectral-refractive transfer: trace once, relight by dot product":
        "render glass fast | glass render without path tracing | relight glass without re-rendering | bake the "
        "refraction once | render is a dot product | no noise glass render",
    "catalog:Grid-free PDE solve on an SDF (Walk on Stars)":
        "walk on spheres | grid free solver | boundary value problem",
    "catalog:Ground-plane depth (forward-looking perspective)":
        "ground plane depth | horizon depth ramp | depth for a road or track scene",
    "catalog:Hadamard codebook (cleanup as one transform)":
        "cleanup without comparing against every codebook entry | structured codebook so cleanup is a transform | "
        "nearest codeword in log time | speed up cleanup when the codebook is huge | maximum likelihood nearest "
        "codeword | green machine decoder | cleanup faster than a matmul",
    "catalog:Holistic lattice cleanup (FHRR resonator factoring of FPE coordinates)":
        "factor a bound product of lattice coordinates | recover integer coordinates from a hypervector | "
        "resonator cleanup to nearest lattice point | decode a fractional-power-encoded position | snap a "
        "holographic coordinate to a lattice",
    "catalog:Holographic drift generative model (HDRIFT: train on points, generate by drift)":
        "train a generative model | generate new samples like my data | make more data like this | conditional "
        "generation by label",
    "catalog:Holographic texture bake (scatter/gather fast path)":
        "scatter gather texture bake",
    "catalog:Honesty & measurement":
        "false discovery rate | proof of structure",
    "catalog:Hybrid body plans (a centaur is a spec, not a code path)":
        "mount a limb on another limb | chimera body plan | arms on a horse",
    "catalog:Hydraulic terrain erosion (droplet simulation)":
        "make procedural terrain look weathered | drainage channels on a landscape",
    "catalog:Identify an element by its properties":
        "identify an element from its properties | find the element that is an inert gas | reverse periodic table "
        "lookup | classify an element by category | reverse periodic table lookup | which element matches these "
        "properties",
    "catalog:Image analysis (classic CV)":
        "detect corners in an image | find lines in an image | cluster images by appearance | image feature "
        "vector | perceptual image descriptor | analyze an image",
    "catalog:Information-rate rendering (shade the news, reproject the rest)":
        "oldest pixel refresh | render fewer pixels",
    "catalog:Insurance profile (does filtering delete the effect?)":
        "is the payoff concentrated in the times I would exclude | where does my profit actually come from | is "
        "this effect insurance | rare event pays for everything | safe to prune this rarely used path | value "
        "concentrated in few events",
    "catalog:Invite guests and share selectively (access control)":
        "who can read | admit a guest",
    "catalog:Iridescent thin-film tint (soap bubble / oil slick)":
        "oil slick sheen | thin film interference | beetle shell film iridescence",
    "catalog:Journal-first documents in the workspace (content-addressed assets + op journals, GC)":
        "content addressed blob in the workspace | op journal instead of pixel snapshots | garbage collect unused "
        "assets | deterministic document replay across apps",
    "catalog:Learning & agents":
        "train a classifier",
    "catalog:Light a render with a real HDRI (.exr in, exact floor irradiance, the map picks the shadow)":
        "use an hdri to light the scene | environment map lighting | image based lighting | openexr environment "
        "map | where is the sun in my hdri | how hard should the shadow be | poly haven hdri",
    "catalog:Live shared workspace (.lews standard): versioned kinds, canonical sections, apps editing together":
        "apps built on lecore work together",
    "catalog:Load an HDRI environment map (.hdr RGBE -> unbounded radiance)":
        "load an hdri environment map | light my scene with a real sky photo | read a radiance hdr file | load a "
        "high dynamic range image | open an hdr file",
    "catalog:Loss space report (where the losses live, per axis, vs its own null)":
        "where do the losses concentrate | loss concentration report | which states lose the money | breakdown of "
        "losses by condition | loss tail heavier than gaussian",
    "catalog:Make a mesh manifold (split non-manifold vertices)":
        "resolve a bowtie vertex | cut non-manifold edges | unfan a vertex",
    "catalog:Make the attached LLM a planner-visible tool":
        "let the planner use the language model | register an llm as a tool | make the model visible to the "
        "planner | llm as a tool | use my model in a plan",
    "catalog:Make water (one-call Gerstner ocean preset)":
        "ocean heightfield generator | one call water",
    "catalog:Mask refraction (2D lens/droplet distortion)":
        "water droplet distortion | glass blob effect | lens distortion from a shape | screen space refraction | "
        "water shimmer on an image",
    "catalog:Massive sharded game world (deterministic migration)":
        "entity migration between shards | world scale simulation | distribute a game world across machines | "
        "open world game backend | seamless world regions",
    "catalog:Memory (cache hierarchy)":
        "backend recomputing backends",
    "catalog:Mesh report (topology scoreboard)":
        "inspect a mesh | quad percentage and valence | is my mesh watertight",
    "catalog:Mixed minerals in one growth: per-seed habits with their own lattices (grow_on habit=list|callable)":
        "different crystal types growing together | mixed habits in a cluster | quartz and calcite on the same "
        "rock | crystal lattices for mixed types | per-seed crystal habit",
    "catalog:Move / rotate / scale an object (and actually render the rotation)":
        "move an object in the scene | rotate an object I already placed | turn a cube 45 degrees | tilt an "
        "object | my rotation is not showing up in the render | orient an object | why did my object not rotate | "
        "position an object in the scene document",
    "catalog:Multi-tone generator (independent, incommensurate frequencies)":
        "sum of sinusoids | multi tone fit | fit multiple sine waves",
    "catalog:Multi-way tensor compression (Tucker / TT)":
        "compress a frame stack | rank gate several compress",
    "catalog:NTT exact integer binding":
        "exact circular convolution with integers | bind two vectors with no rounding error | convolution that is "
        "identical on every machine | integer only binding | bind without floating point | deterministic binding "
        "across cpus | binding with no rounding",
    "catalog:NURBS":
        "evaluate a nurbs patch | exact circle spline",
    "catalog:Navigation & planning":
        "plan a route | solve a maze",
    "catalog:Nebula (volumetric gas & dust)":
        "interstellar dust cloud | make a nebula | gas and dust cloud",
    "catalog:Nested memory library (many knowledge bases in ONE vector, one-unbind queries)":
        "library of memories | memory of memories | query across model shelf | two level lookup one operation | "
        "shelve a trained memory",
    "catalog:Node-graph editor backend":
        "node graph editor | shader node graph | connect nodes with typed sockets | visual node graph | node "
        "graph backend | dataflow node editor | time varying node graph | make geometry react to music | music "
        "reactive visuals | drill down to a setting | list a node's parameters | inspect a node | adjust exact "
        "settings | tweak a node's value",
    "catalog:Object handles over /invoke (name a live object across calls)":
        "put a sphere into the scene document | delete an object from the scene | undo my last scene edit | "
        "reference a returned object in the next call | handle for a non serializable result",
    "catalog:Object snap: midpoint + intersection":
        "snap to midpoint | snap to where lines cross",
    "catalog:One kernel, two runtimes: Zig raymarcher, bit-identical":
        "native sphere trace | compare two renders | one kernel two runtimes | bit identical render",
    "catalog:One pipeline, three bodies (the unification regression)":
        "hybrid body plan test | rig from a point cloud | do all body plans use one code path",
    "catalog:Orient (the front door: full capability access at a model's fingertips)":
        "how do I use lecore | direct me to the right tool | agentic workflow for lecore",
    "catalog:Ouroboros (the closed memory loop: leCore eats the installed model's memory)":
        "feed the model's memory back | manage the installed model's memory | the snake eats its tail | external "
        "memory of the installed model",
    "catalog:Outcome feedback to the attached model (close the one-way seam)":
        "let the model see its mistakes | record what happened to a tool call | does the model learn from results "
        "| one way seam | report the verdict back to the model",
    "catalog:Parametric sky (time of day, sun, moon, stars, high cloud layers)":
        "night sky with stars | cloudy sky with sun shining through | partially cloudy sky | sky sphere "
        "environment",
    "catalog:Parametric surface analysis (curvature + draft)":
        "curvature of a nurbs surface",
    "catalog:Partition byte census (where does my memory file's size GO)":
        "where do the partition bytes go | why is my memory file so big | memory growth rate diagnosis | compress "
        "my saved memory",
    "catalog:Period of a signal (Lomb-Scargle)":
        "orbital period from radial velocity | phase fold a time series",
    "catalog:Physically-based TISSUE materials (organs, bone, fat, skin -- not flat)":
        "organ material chitin deeper",
    "catalog:Pipeline null (did my PROCESSING manufacture the structure?)":
        "run my whole pipeline on surrogates | null for a processing chain | is my smoothing creating the signal "
        "| test the pipeline not just the statistic | did the preprocessing invent this | surrogate through the "
        "same steps | am I fooling myself with resampling | did my pipeline invent the structure",
    "catalog:Points to mesh (isosurface / surface reconstruction)":
        "sdf from a point cloud | point cloud to mesh | extract a surface",
    "catalog:Pose a creature mid-stride, and budget a render before running it":
        "pose a creature | creature mid stride",
    "catalog:Position field (IFAM 4-PoSy lattice remesh)":
        "field aligned lattice | regularise vertex spacing | position field remesh",
    "catalog:Procedural plants & trees (L-system grammar)":
        "grow a tree from rules",
    "catalog:Proportional edit (soft grab with falloff)":
        "grab with falloff",
    "catalog:Quantum statistics (spacing-ratio regime classifier + the Bell verdict)":
        "chsh violation check | quantum correlations test | does my data violate the classical bound",
    "catalog:Quick material ball (plain numbers, no channels)":
        "material ball from numbers | preview roughness and metallic | simple material ball | material editor "
        "preview | try a material without textures | one call material ball",
    "catalog:Recall-budgeted vector index (the forest carries a measured honesty label)":
        "nested descent retrieval | know when it doesn't know | honest approximate search | never silently ship "
        "low recall",
    "catalog:Reclock persistence vs its own null (the manufactured-momentum trap)":
        "honest brick persistence",
    "catalog:Recover a body from a mesh (spine, thickness, inferred tissue)":
        "find the spine of a mesh | recover a rig from a scan | reverse engineer a creature | mesh to rig",
    "catalog:Recursive factoring (past the resonator's cliff)":
        "factor using learned chunks | macro codebook factoring | expand by lookup",
    "catalog:Refine a scene toward a target image (the self-improving loop)":
        "self improving render loop | make my scene look like this picture | close the loop on a render | propose "
        "edits ranked by improvement",
    "catalog:Refine loop (produce / critique / adjust)":
        "produce critique adjust | analysis by synthesis | draft and revise | improve until accepted",
    "catalog:Regenerable audit (store the text, regenerate the vectors)":
        "shrink my saved memory | make the partition small | middle out memory compression | diminishing file "
        "growth",
    "catalog:Render a specimen (adaptive trace + denoise + graded, one call)":
        "render a crystal | one call render | how do I render a gem",
    "catalog:Render quality gate (absolute defect thresholds, not diff-against-last-render)":
        "did my render regress | check a render for terracing | colour fringe on silhouettes",
    "catalog:Render to text from the weights (installed image formation -> PGM)":
        "render from the weights | picture out of the model | pgm from the model",
    "catalog:Reproject a uv map onto changed topology (seam-aware)":
        "reproject uv after decimation | keep uvs through retopo | uvs lost after remesh",
    "catalog:Resource policy (what this process may use)":
        "turn off the gpu | configure resource limits | stop it using all my cores | system configuration "
        "settings | restrict hardware usage",
    "catalog:Resting fills + paper book (passive adverse selection; forward test with gates)":
        "resting order adverse selection | passive fill toxicity | queue position fill model | simulated account "
        "with sleeves and medians | cost of being filled passively",
    "catalog:Roles as powers of one shift (the affordable role machine)":
        "role filler machine | powers of one operator",
    "catalog:Rolling / streaming statistics (causal by construction, exact by default)":
        "causal rolling standard deviation | rolling statistics without look ahead | running percentile of a "
        "stream | trailing drawdown series | ewma volatility series | simple moving average | smooth a series "
        "with a sliding window",
    "catalog:Roofline predictor (price a byte-moving change before you build it)":
        "how long will it take to stream this much data | predict memory bandwidth time | is my change worth it "
        "before i build it | tokens per second from bandwidth",
    "catalog:Run any faculty as a background job":
        "run a faculty asynchronously | start a job",
    "catalog:Runtime-discovery abstention (does it abandon AFTER starting?)":
        "does it give up when a tool breaks | did it notice the plan stopped working",
    "catalog:SDF primitive pack":
        "basic sdf shapes | add a primitive to a scene | sdf building blocks",
    "catalog:Save and load a retrieval index (one format, disk and browser)":
        "save an index to disk and load it back | indexeddb local storage for leCore",
    "catalog:Scaffold meshing (a cage on the skeleton, projected onto the field)":
        "my limbs look lumpy | beading on thin limbs",
    "catalog:Scale (distribute)":
        "map reduce sparsefield pieces | parallel sparsefield pieces",
    "catalog:Scatter bake & level of detail (measured)":
        "level of detail for grass | cache a scatter | reduce grass in the distance | bake the grass | too many "
        "blades",
    "catalog:Scatter meshes over a surface (grass, rocks, plants)":
        "put grass on my terrain mesh | instance a model many times over a mesh | barnacles on a hull | populate "
        "a surface",
    "catalog:Screen a battery of detectors (honesty gates inside the loop)":
        "battery of checks as one program | test many hypotheses with fdr built in | committee of detectors that "
        "refuses to overfit | screen candidates honestly | combine signals with survival gates | empty committee "
        "as a result | battery screening with honesty gates",
    "catalog:See what the mantis sees (false colour)":
        "visualize polarization as color | map invisible channels to rgb | see ultraviolet as visible color | "
        "polarization angle to hue | wavelength to rgb",
    "catalog:Segment a photo into object regions (demux)":
        "segment an image | segment a photo into objects | split an image into regions | connected components of "
        "an image",
    "catalog:Semantic action menu coverage (verb tags)":
        "semantic tag coverage | action menu coverage | how many capabilities are tagged",
    "catalog:Serve leCore as a tool (/tools + /invoke)":
        "serve as a tool | let an agent use leCore",
    "catalog:Settle-gated simulation runner (pay for dynamics, not equilibrium)":
        "stop simulating when settled | speed up my simulation | graphics optimization pass",
    "catalog:Shader-native atom families (a vocabulary that is a FUNCTION, not a table)":
        "generate a codebook instead of storing it | bind exactly without an FFT | role filler record from names "
        "| zero byte vocabulary | recompute the same atom on the GPU | browser friendly hypervector atoms | one "
        "place to normalise a term | same term id in both search arms | factor a complex product back into its "
        "parts",
    "catalog:Shadow / visibility (domain)":
        "ambient occlusion visibility strategies",
    "catalog:Signal & spectral":
        "fft flatness doppler | detect a signal | bandwidth flatness doppler",
    "catalog:Skin a skeleton (B-Mesh base mesh)":
        "skin a skeleton | tube mesh from edges with radii | blockout mesh from a skeleton",
    "catalog:Sky observation (cube + world axes)":
        "world coordinate axes | pixel to sky coordinate | load a telescope cube | observation with RA Dec freq | "
        "ingest a sky map",
    "catalog:Spatial memory (position hypervectors: closest-point as associative recall)":
        "position keyed memory | encode 3d points as hypervectors | look up what is near a location",
    "catalog:Spin up local worker processes (parallel execution)":
        "spin up another instance | start a second worker | use more cores | run work in parallel across "
        "processes | parallel execution on one machine | balance load across instances | make it use all my cpus "
        "| local process pool",
    "catalog:Splat aniso-refine (re-enable)":
        "coarse first splat",
    "catalog:Stage contracts for the render pipeline (end-to-end missed four)":
        "check my render pipeline is wired right | which stage of the render is broken | verify the renderer end "
        "to end",
    "catalog:Standard agent surface for an app built on leCore (tools, invoke, mind, engine, events, presence)":
        "tool manifest from flask routes | which engine build am I running | X-User X-Client identity headers | "
        "make my app a good lecore citizen",
    "catalog:State demand meter (how much memory does this stream need)":
        "how much state does this stream need | bond dimension of a process | entropy rate of a signal | memory "
        "demand before allocating",
    "catalog:Stateless coordinate-keyed randomness (hash_unit)":
        "no seed coordination | farm parallel sampling | random number per thread without a seed stream | per "
        "thread rng",
    "catalog:Stock a part library (why your creature parts do not render)":
        "my creature parts do not show up | place_parts returns nothing | how do i add geometry to a part library",
    "catalog:Store a multi-way array (tensor-train file)":
        "save a volume",
    "catalog:Stream to OBS (browser-source capture profile)":
        "add leos to obs | obs browser source settings | streaming setup for obs | transparent background for "
        "streaming | record or stream the canvas | use this in my stream",
    "catalog:Streaming meters (live convergence guard and entropy, O(1) per sample)":
        "online convergence check | real time stream analysis | audio block meter",
    "catalog:Studio lighting rig (three-point as an environment, for the path tracer)":
        "key fill and rim light | soft lighting for a product shot | environment lighting for a path tracer",
    "catalog:Superposed key-value memory (capacity law + allocator + gated resonator decode)":
        "how many pairs fit in a vector | associative memory size | interference cancellation recall",
    "catalog:Surface-route retopology (field-aligned quads, silhouette-safe by construction)":
        "retopologize a scan | quad remesh a photogrammetry scan",
    "catalog:Swarm use-case doors (service escalation, shared codebase sync, bus roles)":
        "escalate to a human and remember the answer | shared understanding of the codebase across agents | sync "
        "the swarm's knowledge of the code | a lab of agents on one bus and one memory",
    "catalog:Table-to-analyst bridge (a column IS a series)":
        "regimes in a database column | run analytics on my table",
    "catalog:Tetrahedralize a point set with PROVED topology (limb-connection certificates)":
        "decimate without breaking topology | volumetric mesh from points | is my limb connected",
    "catalog:Texture-preserving mesh repair & decimation (attribute-aware weld)":
        "keep uvs when decimating a mesh | mesh optimization loses texture coordinates | preserve texture through "
        "remesh",
    "catalog:Textured object render (paint composed maps)":
        "paint texture on object | map onto object",
    "catalog:The HRR algebra for builders (bind / bundle / unbind / cosine / nearest / derived_atom)":
        "the hrr algebra | holographic reduced representation | circular convolution bind | deterministic atom "
        "for a symbol | cosine similarity of hypervectors | how do i build on the holographic framework",
    "catalog:The Holographic RNN (measure -> identify | price | refuse, with provenance)":
        "recurrent model with provenance | should I fit a generator or a memory | trajectory classifier with "
        "signatures | should I fit a generator or a memory | generator versus memory decision",
    "catalog:The SDF DSL, described well enough to write one":
        "what nodes does the sdf dsl have | shape language reference | union two shapes together | subtract one "
        "shape from another",
    "catalog:The above/below sweep (is every declared capability reachable at every layer?)":
        "find capabilities that exist in the engine but nowhere else | audit the layers of the stack for gaps | "
        "does the catalog promise anything it cannot deliver",
    "catalog:The inner eye's 2D toolset (image ops as installed chain steps)":
        "blur inside the weights | flip is a permutation",
    "catalog:The showcase (runnable proof of what makes this engine different)":
        "what makes this project special | show me what it can do | why is this different | prove the claims",
    "catalog:The time machine (unitary dynamics: reversible, random-access, superposable time)":
        "run the simulation backwards | time travel state | reverse the dynamics | many simulations one vector | "
        "undo n steps",
    "catalog:Token sampling (temperature + nucleus)":
        "sample stochastic draw | sample the next token | instead of always picking the best | sample from a dict "
        "of scores",
    "catalog:Topology gate (reject remeshes that punch holes or shatter components)":
        "genus and boundary loop check | detect mesh fragmentation",
    "catalog:Trace energy partition (the saturation ledger: signal / crosstalk / damage)":
        "how much of this bundle is signal | signal versus crosstalk fraction",
    "catalog:Trace streamlines (field -> curves)":
        "trace a direction field | guide curves from a cross field",
    "catalog:Transform (warp)":
        "bind representations quaternions | permute representations quaternions",
    "catalog:Trimmed surface":
        "trim a surface | surface with a hole | trimmed nurbs face | bounded surface patch",
    "catalog:True import footprint of an entry point (bundler's answer)":
        "what modules does this actually need at runtime | minimal dependency set for bundling a subset | true "
        "dependency footprint | bundle a subset of the engine",
    "catalog:Tunnelling & CCD (speculative margins, conservative advancement)":
        "tunneling passing bodies | time of impact | when will my object hit the ground | grow a collider by a "
        "small amount | offset an sdf | bullet through paper | prevent objects passing through each other",
    "catalog:VSA cleanup on ANY GPU (matvec + argmax, fused)":
        "cleanup on the gpu | matrix times vector on the gpu | codebook similarity on any gpu | clean up many "
        "cues at once",
    "catalog:VSA load-bearing audit (the ablation table)":
        "is vsa load bearing here | honest baseline comparison | which subsystems need vsa | vsa vs simple "
        "baseline",
    "catalog:VSA programs as DB objects":
        "run program whitelisted procedures",
    "catalog:Validated termination: hold a step's RETURN to a contract, retry bounded (the exit gate)":
        "retry a tool call when it returns nothing useful | typed return validation with retry | make the agent "
        "prove the step finished | bounded retry with the error fed back | validate what a step returned before "
        "moving on",
    "catalog:Verified GLSL kernels (shader source with the number that verified it)":
        "diffusion step shader | which shaders have been tested",
    "catalog:Video drift (train on short clips, generate coherent motion)":
        "generate video like these clips | train on my short clips | video texture generation",
    "catalog:View transform (linear render -> a display image)":
        "my render is blown out | the image is too bright | tonemap an hdr render | aces filmic view transform | "
        "convert a linear render to a display image | exposure for a render | the render looks flat and grey | "
        "make the render look cinematic",
    "catalog:Viscera that are separate organs (disjoint, with identity)":
        "why do my organs look like one blob | label each organ separately | anatomically separate viscera",
    "catalog:Voxelization":
        "voxelize a mesh | occupancy grid from a mesh | inside outside mesh | voxel point cloud | mesh to voxels "
        "to mesh | rasterize a mesh",
    "catalog:What does this actually return? (the return-shape probe)":
        "probe a method before calling it | probe a faculty signature | how do i call this faculty | what "
        "arguments does this take | inspect a return value | why is this returning the wrong thing | attribute "
        "error on a returned object | is this a dict or an object",
    "catalog:Who is in the workspace (cross-app presence, host, notes, long-poll, open an app's .lews file)":
        "who else is editing this workspace | presence across apps | show other users cursors tools | host of the "
        "session | import lestudio workspace",
    "catalog:Will compression pay? (area law vs volume law)":
        "is this compressible",
    "catalog:aces_tonemap":
        "tonemap an hdr image",
    "catalog:add_caustics":
        "focused light patterns",
    "catalog:analytic_signal":
        "sign as rotation | value as rotation | envelope of a signal | represent negative as rotation | circle "
        "encoding of a value",
    "catalog:analyze_axes":
        "carrier vs content | which axis is the carrier | which axis is boring | index or bind | elevate the "
        "boring dimension | which dimension to fold in",
    "catalog:applications":
        "a library of runnable applications I can try | try a sample application | what can this engine actually "
        "do, show me",
    "catalog:ascii_animate":
        "animate in the terminal | play frames as ascii | animated ascii art | render a sequence of frames to "
        "text",
    "catalog:audio_param_bus":
        "audio reactive parameters | onset to parameter | audio param bus | frequency bands over time | make a "
        "demo react to music",
    "catalog:bank_or_formula":
        "should I cache or recompute | bank versus formula decision | when to store versus recompute",
    "catalog:bios_boot":
        "boot the mind | os for the llm",
    "catalog:boot":
        "boot the substrate | run startup self checks | substrate self check",
    "catalog:catalog_families":
        "unresolved capability families | what folder does this tool live in",
    "catalog:central_mass_from_orbit":
        "kepler third law | mass of a star from an orbit | weigh a star | astronomy mass estimate | semi major "
        "axis period",
    "catalog:codebase_diagram":
        "generate a codebase diagram | mermaid diagram of code | architecture diagram from source | graphviz of "
        "the repo",
    "catalog:comparability_cost":
        "cost of binding an axis | binding destroys similarity | measure binding cost",
    "catalog:creature":
        "spore creature editor | spine with limbs",
    "catalog:decision_outcome":
        "close the loop on a decision by id | record what was used | outcome feedback by id | save my decisions "
        "to memory | remember past decisions across sessions | generate the prompt for the model end from the "
        "schema | answer a repeated request from experience",
    "catalog:decompose_piecewise":
        "fit a law per regime | signal with multiple regimes | decompose in pieces",
    "catalog:delegation_drift":
        "fix a wrapper that lost an argument | check my faculty matches its module function",
    "catalog:depth_from_image":
        "estimate depth from a photo | shape from shading | depth map from an image | guess depth from a picture "
        "| single image depth estimation",
    "catalog:disable_cold_storage":
        "keep all rows hot | turn off query tiering",
    "catalog:dispersive_render":
        "dispersion glass material | glass bsdf dispersion | abbe number glass shader | prism glass shader",
    "catalog:escape_time":
        "escape time fractal | 2d fractal escape count | complex z^2+c fractal",
    "catalog:explain_similarity":
        "why are these two similar | compare two records role by role",
    "catalog:extend_generator":
        "evaluate a generator at future time",
    "catalog:feedback_and_deep_zoom":
        "infinite zoom demo | as above so below, same structure at every scale | the same operator on a field and "
        "on a sequence",
    "catalog:field_displace":
        "per face modifier from a texture | drive geometry from a field | vertex displacement from an sdf | "
        "mandelbulb modifier on a mesh | texture masked displacement | apply a fractal modifier to geometry",
    "catalog:fit_deterministic":
        "fit a generator to a signal | reverse engineer a signal",
    "catalog:fit_primitives":
        "approximate a shape with primitives | fit sdf primitives to a shape | sphere box capsule fit",
    "catalog:fit_shape":
        "fit a shape and get shadertoy | match a model to a fractal | closest procedural fit | fit a texture to a "
        "formula | represent a shape with an equation",
    "catalog:fold_fit":
        "fit a fractal recipe to a shape | recover fold parameters | inverse fractal problem",
    "catalog:four_surface_demo":
        "one kernel four surfaces | all backends of a scene | scene to every format",
    "catalog:fuse_rankings":
        "combine ranked lists | merge two rankings | fuse dense and sparse retrieval | hybrid search fusion | "
        "blend search results by rank",
    "catalog:guide_structure":
        "iterate a projection | project onto constraints | reach a goal under constraints",
    "catalog:holographic_catalog":
        "duplicate building register",
    "catalog:holographic_rayindex":
        "pixels touches distinct",
    "catalog:iaaft_surrogate":
        "exact spectrum and distribution null",
    "catalog:identify_dynamics":
        "mass from trajectory and force | fit equation of motion | mass ratio from collision | damping and "
        "stiffness from data | can i get mass from a trajectory",
    "catalog:identify_level":
        "what level of abstraction is this | classify a corpus | does this data have hierarchy | which lens fits "
        "this data",
    "catalog:ifs_generate":
        "generate a fern | make a fractal tree | chaos game fractal | affine ifs attractor | draw a fern",
    "catalog:image_to_3d":
        "3d gaussians from an image | picture to 3d points | gaussian splatting from a photo | 3d from a single "
        "photo | image to point cloud | 3d from one image | turn a photo into 3d | photo to 3d model",
    "catalog:lews_section":
        "make a container section | build a lews section | stamp a schema version | add a section to a lews file",
    "catalog:match_prototype":
        "classify without a schema | nearest prototype match | match a blend to a class | which class does this "
        "blend fit",
    "catalog:match_record":
        "match by structured record | structure-aware nearest match | rank candidates by their attributes | which "
        "class does this record fit | match physics regime market event astronomy source by structure",
    "catalog:merge_drift":
        "reconcile drifted replicas",
    "catalog:mesh_auto_seam":
        "automatically place uv seams | mark seams by curvature | seam along sharp edges | where to place uv "
        "seams",
    "catalog:mesh_rip_vertex":
        "rip a vertex | tear a mesh at a vertex | split a shared vertex | rip vertices apart",
    "catalog:milk_parse":
        "parse a milkdrop preset | read a .milk file | milkdrop preset reader | run milkdrop equations",
    "catalog:mutual_information":
        "how much does x tell me about y | dependence between two variables | are two signals related | is my z "
        "score sample inflated",
    "catalog:not_null":
        "filter missing values | rows with a value | is not null",
    "catalog:orbit_trap_render":
        "quilez orbit trap look | render with orbit traps | iq fractal colors",
    "catalog:packet_demux":
        "variable length bursts",
    "catalog:perfect_recall_index":
        "perfect recall search | guaranteed no false negatives index | bloom filter membership over documents | "
        "unlimited corpus exact retrieval",
    "catalog:plan_change":
        "which module should a new feature go in | build loop steps with done when",
    "catalog:plan_from_request":
        "break a request into steps",
    "catalog:query_fuzzy":
        "similar-value scene query",
    "catalog:redo_stack":
        "what can be redone",
    "catalog:reflex_retile":
        "retile the reflex trace | reflex trace past capacity | the reflex stopped firing after loading memory | "
        "split the experience trace into more tiles",
    "catalog:reject_outliers":
        "reject outlier samples | bright dots in a path trace | robust mean across buckets | salt and pepper "
        "noise in a render",
    "catalog:resolve_reference":
        "what did 'it' refer to | anaphora over results",
    "catalog:retrieval_dispatch":
        "adaptive search cascade | pick the right retrieval method | search that stops when the answer is proven "
        "| hybrid retrieval without scoring everything | route between dense and lexical search | denoise a "
        "search shortlist | choose between keyword and vector search",
    "catalog:route_structured":
        "route by structure not keywords | holographic role router | structured routing by binding | which module "
        "by request structure",
    "catalog:sandbox_run":
        "run code in a sandbox | execute python safely | test a snippet with limits | safe code execution | run "
        "this script isolated",
    "catalog:screen_ray":
        "screen to world ray | cursor to ray | unproject a screen point | ray from a screen coordinate",
    "catalog:select_in_box":
        "region select vertices | rubber band select | select points in a box",
    "catalog:select_symmetric":
        "select the other side too | select symmetric vertices",
    "catalog:sellmeier_ior":
        "index of refraction for a wavelength | how much does glass bend blue light | refractive index of BK7 | "
        "abbe number of a glass | pick a glass for a rainbow",
    "catalog:similar_to":
        "fuzzy where clause",
    "catalog:snap_to_vertices":
        "snap to nearest vertex | snap a vertex to another",
    "catalog:snap_transform_delta":
        "snap a move to the grid | constrain a move to a snap target | snap the gizmo delta",
    "catalog:sphere_trace_trapped":
        "closest approach along a ray | trap distance per pixel",
    "catalog:systemone_batch_fdr":
        "false discovery rate over a batch of decisions | control how many wrong accepts in a batch",
    "catalog:systemone_lint":
        "lint a decision schema before asking | check my examples are balanced | which scorer should I use | why "
        "did my typed decision abstain on everything | one observation per state",
    "catalog:systemone_stream":
        "prequential evaluation of decisions | close the decision loop | online prototype update from mistakes | "
        "escalate with schema enforced",
    "catalog:transform_selection":
        "translate rotate scale a selection | move a selection | transform in a space | proportional edit "
        "transform",
    "catalog:typed":
        "classify this into one of a few categories | make a decision with a few examples | one line typed "
        "decision",
    "catalog:vsa_region":
        "spherical region algebra | combine regions of space",
    "catalog:workspace_manager":
        "save a workspace | save my work | persist a scene | restore a workspace | export a scene | workspace "
        "save and load | checkpoint a scene | restore a checkpoint",
    "catalog:write_multichannel":
        "embed data in weights | multichannel steganographic write",
}

def examples():
    """[(text, type, option)] in option order (sorted keys), example order within an option kept: the typed lists,
    then the engine snapshot (all ordinary)."""
    out = []
    for option in sorted(GUARD_EXAMPLES):
        typ, blob = GUARD_EXAMPLES[option]
        out += [(t.strip(), typ, option) for t in blob.split(" | ") if t.strip()]
    for option in sorted(ENGINE_EXAMPLES):
        out += [(t.strip(), "ordinary", option) for t in ENGINE_EXAMPLES[option].split(" | ") if t.strip()]
    return out


def _selftest():
    """The data contract the typed guard relies on: every option has a known type and at least one example, the
    four types are all present, and no example carries a credential VALUE (questions only -- the pattern layer
    must not refuse any line here, or the guard would be built from text it would itself refuse to learn)."""
    from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
    rows = examples()
    types = {typ for _, typ, _ in rows}
    assert types == {"credential", "live_value", "unclear", "ordinary"}, types
    assert all(t for t, _, _ in rows), "an empty example"
    for option, (typ, blob) in GUARD_EXAMPLES.items():
        assert typ in types and blob.strip(), option
    leaks = [t for t, _, _ in rows if sensitive_reason(t)]
    assert not leaks, "examples that look like a secret VALUE: %r" % leaks[:3]
    return "ok"


if __name__ == "__main__":
    print(_selftest())
