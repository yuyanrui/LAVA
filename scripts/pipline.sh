set -x

DATASET=warsaw
query=('bus with multi-section design'
 'white van'
 'bus with single-section design'
 'tram with yellow and red coloration')
DATASET=$1
predicate=$2

sublist='test'
q=${predicate}
bash scripts/train_prompt_full/train_full_object.sh 1 ${DATASET} "${q}"
bash scripts/segment_localization/run_localization.sh 1 ${DATASET} "${q}" ${sublist}
bash scripts/lava/run_query.sh 1 ${DATASET} "${q}" '${q}' ${sublist}