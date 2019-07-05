experiments = {...
  GlmSelectAicAbsNoFilterNull();
  GlmSelectAicAbsNoFilterBw();
  GlmSelectAicAbsNoFilterLinear();
  GlmSelectBicAbsNoFilterNull();
  GlmSelectBicAbsNoFilterBw();
  GlmSelectBicAbsNoFilterLinear();
  GlmSelectAicAbsFilterNull();
  GlmSelectAicAbsFilterBw();
  GlmSelectAicAbsFilterLinear();
  GlmSelectBicAbsFilterNull();
  GlmSelectBicAbsFilterBw();
  GlmSelectBicAbsFilterLinear();
};

parfor i = 1:numel(experiments)
  experiments{i}.run();
end