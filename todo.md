# Fases
- 1
    - Verloop
        - (Enquete)
        - Ingeven nickname
        - Calibratie(/"training")
            - 30s afleiding
            - 30s ontspanning
            - 30s ogen dicht
            - casual spelen
            - intens spelen
        - Spelen
            - Beperkt aantal objectives? -> Alleen built-in!
            - Volledig statisch (zelfs predetermined drops?) -> vaste diff.
            - Gerandomiseerde volgorde objectives? -> NVT
        - Vrij spel? -> Nee.
    - Verworven data
        - Enquete
        - Calibratie
        - Continue speel data? -> Nee.
        - Evaluaties? -> Nee.
        - Free time data? -> Nee.
- 2
    - Verloop
        - Training tot volledig gecalibreerd
        - Spelen met vooraf vastgelegde difficulty curves
        - Evaluaties na spelen
        - ...?
    - Verworven data
        - difficulty <-> "toestand" <-> evaluatie
        - training encoder
        - training classifier


# Essentieel
- Verwerking enquete (fase 1)
    - Google docs -> JSON
- Simulator (fase 1? - ) *tegen 14/10*
    - Aantal matches vinden
    - Mogelijke puzzel objectives vinden + grading
    - Actie objectives grading (curve?)
    - Bepaling drops om moeilijkheidsgraad te beinvloeden?
+ Logging (fase 1 - ) 
    + Toevoegen!
    + Integreer bw + spel data (+ debugging, ...?)
- Stages (fase 1 - )
    + Ingeven nickname
    + Calibratie (fase 1)
    - (Spel)
    - Evaluatie? (fase 1? 2?) *tegen 9/10*
- Objectives (fase 2 - )
    + Oud systeem er uit smijten *tegen 14/10*
    - Events bij maken die kunnen geinterpreteerd worden door nieuwe objectives *tegen 14/10*
    - Aantal objectives maken *tegen 21/10*
        - Actie
            - x punten in y tijd *tegen 16/10*
            - x matches in y tijd *tegen 14/10*
            - x 4-matches in y tijd *tegen 16/10*
            - (nieuwe mechanismes)
        - Puzzel
            - cascade van x punten (in lange tijd) *tegen 21/10*
            - x matches van zelfde soort na elkaar *tegen 21/10*
            - specifieke opeenvolging matches? *tegen 21/10*
            - (nieuwe mechanismes)
    - Moeilijkheidsgraad systeem
        + Basisimplementatie actie
        - Calibreren actie *requires: Objectives, Simulator*
        - Basisimplementatie puzzel *requires: Objectives*
        - Calibreren puzzel *requires: Objectives, Simulator*
        
- ML (fase 2 - )
    - SDA
        - Implementeren (theano? caffe? ...) *tegen 9/10*
        - Trainen op fase 1 data (compleet) *tegen 18/10*
        - Trainen op fase 2 data *tegen 28/10*
        - ...
    - RNN
        - Implementeren (...) *tegen 23/10*
        - Trainen op fase 1 data (Calibratie, evaluatie?) *tegen 28/10*
        - Trainen op fase 2 data (Calibratie, evaluatie?,  ...) *tegen 4/11*
    - RL
        - Implementeren
        - Trainen op fase 2 data
        - Trainen op fase 3 data
        - Validatie (fase 4)

# Wishlist
- Additionele mechanismes
    - Drop to bottom
    - Bommen
    - Timed fruit?
    - (zie Candy Crush / Bejeweled?)
