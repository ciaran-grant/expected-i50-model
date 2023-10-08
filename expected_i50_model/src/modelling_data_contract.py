from dataclasses import dataclass

@dataclass
class ModellingDataContract:
    """ Holds details for defining modelling terms in a single place.
    """
    
    ID_COL = "Match_ID"
    RESPONSE = "Inside50"
    TRAIN_TEST_SPLIT_COL = "ModellingFilter"

    feature_list = [
        'quarter_a0',
        'quarter_seconds_a0',
        'overall_seconds_a0',
        'start_x_a0',
        'start_y_a0',
        'end_x_a0',
        'end_y_a0',
        'dx_a0',
        'dy_a0',
        'movement_a0',
        'outcome_effective_a1',
        'outcome_ineffective_a1',
        'outcome_clanger_a1',
        'quarter_a1',
        'quarter_seconds_a1',
        'overall_seconds_a1',
        'start_x_a1',
        'start_y_a1',
        'end_x_a1',
        'end_y_a1',
        'dx_a1',
        'dy_a1',
        'movement_a1',
        'outcome_effective_a2',
        'outcome_ineffective_a2',
        'outcome_clanger_a2',
        'quarter_a2',
        'quarter_seconds_a2',
        'overall_seconds_a2',
        'start_x_a2',
        'start_y_a2',
        'end_x_a2',
        'end_y_a2',
        'dx_a2',
        'dy_a2',
        'movement_a2',
        'team_1',
        'team_2',
        'time_delta1',
        'time_delta2',
        'dx_a01',
        'dy_a01',
        'move_a01',
        'dx_a02',
        'dy_a02',
        'move_a02'
    ]
    monotone_constraints = {}

    outcome_types = [
        'effective',
        'ineffective',
        'clanger'
    ]