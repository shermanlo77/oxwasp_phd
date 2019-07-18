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

for i = 1:numel(experiments)
  experiments{i}.run();
  experiments{i}.printResults();
end
