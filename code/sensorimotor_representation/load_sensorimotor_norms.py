import os
import pandas as pd


def load_sensorimotor_norms(path):
    sensorimotor_norms = pd.read_csv(os.path.join(path, 'Lancaster_sensorimotor_norms_for_39707_words.csv'))

    sensorimotor_norms = sensorimotor_norms[['Word', 'Auditory.mean', 'Gustatory.mean', 'Haptic.mean',
                                             'Interoceptive.mean', 'Olfactory.mean', 'Visual.mean',
                                             'Foot_leg.mean', 'Hand_arm.mean', 'Head.mean', 'Mouth.mean',
                                             'Torso.mean']]

    sensorimotor_norms = sensorimotor_norms.rename(columns={"Auditory.mean": "Auditory",
                                                            "Gustatory.mean": "Gustatory",
                                                            "Haptic.mean": "Haptic",
                                                            "Interoceptive.mean": "Interoceptive",
                                                            "Olfactory.mean": "Olfactory",
                                                            "Visual.mean": "Visual",
                                                            'Torso.mean': 'Torso',
                                                            'Mouth.mean': 'Mouth', 'Head.mean': 'Head',
                                                            'Foot_leg.mean': 'Foot_leg',
                                                            'Hand_arm.mean': 'Hand_arm'})

    sensorimotor_norms['Word'] = sensorimotor_norms['Word'].apply(lambda x: x.lower())
    sensorimotor_norms = sensorimotor_norms.set_index(['Word'])
    sensorimotor_norms = sensorimotor_norms.to_dict('index')

    return sensorimotor_norms
