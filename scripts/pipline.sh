set -x

DATASET=warsaw
query=('bus with multi-section design'
 'white van'
 'bus with single-section design'
 'tram with yellow and red coloration')

# DATASET=caldot1
# query=('Black sedan' 'SUV with roof racks' 'Pickup truck with a flatbed')

# DATASET=caldot2
# query=('semi-truck with a trailer' 'Black SUV' 'black pickup truck')

# DATASET=amsterdam
# query=('White boat with black roof' 'Boat with a long mast and sails' 'Black bicycle' 'White van')

# DATASET=jackson
# query=('SUV without roof rack' 'SUV with roof rack' 'Van with a sliding passenger door' 'white semi-truck with blue and red details')

# DATASET=shibuya
# query=('Bus with single-section design' 'White truck with black lettering' 'Multi-section articulated bus')

for sublist in 'test'
do
for index in 1 2 3
do
    q=${query[index]}
    bash scripts/train_prompt_full/train_full_object.sh 1 ${DATASET} "${q}"
    bash scripts/segment_localization/run_localization.sh 1 ${DATASET} "${q}" ${sublist}
    bash scripts/lava/run_query.sh 1 ${DATASET} "${q}" '${q}' ${sublist}
    bash scripts/baseline.sh 1 ${DATASET} "${q}" ${sublist}
done
done