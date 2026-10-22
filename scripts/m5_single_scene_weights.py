#!/usr/bin/env python3
"""Give precise descent, closure and lift rows more sampling probability."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    args=parser.parse_args()
    schema=json.loads((args.dataset/'schema.json').read_text())
    assert schema['collection']['protocol']=='single_scene_first_attempt_v1'
    split=json.loads((args.raw/'split_manifest.json').read_text())['train']
    weights=[];counts=Counter();mass=Counter()
    for entry in split:
        rows=[json.loads(s) for s in (args.raw/f"episode_{entry['source_episode_id']:03d}/samples.jsonl").read_text().splitlines()]
        for row in rows:
            weight=3. if row['stage'] in ('descend','grasp','lift') else 1.
            weights.append(weight);counts[row['stage']]+=1;mass[row['stage']]+=weight
    result={'repo_id':schema['repo_id'],
            'normalization_sha256':hashlib.sha256((args.dataset/'normalization.json').read_bytes()).hexdigest(),
            'scope':'3x sampling for descent, grasp and lift; other rows 1x. All target actions and training-only normalization unchanged.',
            'stage_rows':dict(counts),'expected_stage_probability':{k:v/sum(weights) for k,v in mass.items()},'weights':weights}
    (args.raw/'sampling_weights.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='weights'},indent=2))
