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

  experiments{i}.run();
end
