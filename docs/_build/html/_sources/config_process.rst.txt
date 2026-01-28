Config Processing From Python
=============================

It is also possible to run the config processing from a Python script, rather than using
`the galsim executable`.  An example of this can be found in
:gh-link:`demo8 <examples/demo8.py>`.

Running the Whole Script
------------------------

The following functions are relevant to running the whole config script from Python:

.. autofunction:: galsim.config.ReadConfig

.. autofunction:: galsim.config.CopyConfig

.. autofunction:: galsim.config.ImportModules

.. autofunction:: galsim.config.ProcessTemplate

.. autofunction:: galsim.config.ProcessAllTemplates

.. autofunction:: galsim.config.Process


Building Files
--------------

The following functions are relevant to building one or more files as specified by a config
dict:

.. autofunction:: galsim.config.BuildFiles

.. autofunction:: galsim.config.BuildFile

.. autofunction:: galsim.config.GetNFiles

.. autofunction:: galsim.config.GetNImagesForFile

.. autofunction:: galsim.config.GetNObjForFile

.. autofunction:: galsim.config.SetupConfigFileNum


Building Images
---------------

The following functions are relevant to building one or more images as specified by a config
dict:

.. autofunction:: galsim.config.BuildImages

.. autofunction:: galsim.config.BuildImage

.. autofunction:: galsim.config.SetupConfigImageSize

.. autofunction:: galsim.config.SetupConfigImageNum

.. autofunction:: galsim.config.GetNObjForImage

.. autofunction:: galsim.config.FlattenNoiseVariance

.. autofunction:: galsim.config.BuildWCS

.. autofunction:: galsim.config.AddSky

.. autofunction:: galsim.config.AddNoise

.. autofunction:: galsim.config.CalculateNoiseVariance

.. autofunction:: galsim.config.AddNoiseVariance

.. autofunction:: galsim.config.GetSky


Building Stamps
---------------

The following functions are relevant to building one or more stamps as specified by a config
dict:

.. autofunction:: galsim.config.BuildStamps

.. autofunction:: galsim.config.BuildStamp

.. autofunction:: galsim.config.SetupConfigStampSize

.. autofunction:: galsim.config.SetupConfigObjNum

.. autofunction:: galsim.config.DrawBasic


Building Objects
----------------

The following functions are relevant to building individual objects as specified by a config
dict:

.. autofunction:: galsim.config.BuildGSObject

.. autofunction:: galsim.config.UpdateGSParams

.. autofunction:: galsim.config.TransformObject

.. autoclass:: galsim.config.SkipThisObject


Generating Values
-----------------

The following functions are relevant to generating and accessing individual values as specified by
a config dict:

.. autofunction:: galsim.config.ParseValue

.. autofunction:: galsim.config.GetCurrentValue

.. autofunction:: galsim.config.EvaluateCurrentValue

.. autofunction:: galsim.config.SetDefaultIndex

.. autofunction:: galsim.config.CheckAllParams

.. autofunction:: galsim.config.GetAllParams

.. autofunction:: galsim.config.ParseWorldPos


Using Input Fields
------------------

The following functions are relevant to processing and using input fields in a config dict:

.. autofunction:: galsim.config.ProcessInput

.. autofunction:: galsim.config.ProcessInputNObjects

.. autofunction:: galsim.config.SetupInput

.. autofunction:: galsim.config.SetupInputsForImage

.. autofunction:: galsim.config.GetInputObj


Processing Extra Outputs
------------------------

The following functions are relevant to processing extra output fields in a config dict:

.. autofunction:: galsim.config.SetupExtraOutput

.. autofunction:: galsim.config.SetupExtraOutputsForImage

.. autofunction:: galsim.config.ProcessExtraOutputsForStamp

.. autofunction:: galsim.config.ProcessExtraOutputsForImage

.. autofunction:: galsim.config.WriteExtraOutputs

.. autofunction:: galsim.config.AddExtraOutputHDUs

.. autofunction:: galsim.config.CheckNoExtraOutputHDUs

.. autofunction:: galsim.config.GetFinalExtraOutput


Config Utilities
----------------

The following functions are used internally by the various ``galsim.config`` functions,
but they might be useful for some users.

.. autoclass:: galsim.config.LoggerWrapper
    :members:

.. autofunction:: galsim.config.ReadYaml

.. autofunction:: galsim.config.ReadJson

.. autofunction:: galsim.config.MergeConfig

.. autofunction:: galsim.config.ConvertNones

.. autofunction:: galsim.config.RemoveCurrent

.. autofunction:: galsim.config.GetLoggerProxy

.. autofunction:: galsim.config.UpdateNProc

.. autofunction:: galsim.config.ParseRandomSeed

.. autofunction:: galsim.config.PropagateIndexKeyRNGNum

.. autofunction:: galsim.config.SetupConfigRNG

.. autofunction:: galsim.config.ParseExtendedKey

.. autofunction:: galsim.config.GetFromConfig

.. autofunction:: galsim.config.SetInConfig

.. autofunction:: galsim.config.UpdateConfig

.. autofunction:: galsim.config.MultiProcess

.. autofunction:: galsim.config.GetIndex

.. autofunction:: galsim.config.GetRNG

.. autofunction:: galsim.config.CleanConfig

.. autofunction:: galsim.config.SetDefaultExt

.. autofunction:: galsim.config.RetryIO

.. autofunction:: galsim.config.MakeImageTasks

.. autofunction:: galsim.config.MakeStampTasks

.. autofunction:: galsim.config.RegisterInputConnectedType
