import glob
head = """<!doctype html>
<html>

<head>
    <meta charset="utf-8">
    <title>NASTAR Demo</title>
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.1/css/bootstrap.min.css" integrity="sha384-VCmXjywReHh4PwowAiWNagnWcLhlEJLA5buUprzK8rxFgeH0kww/aWY76TfkUoSX" crossorigin="anonymous">
    <link rel="stylesheet" href="style.css">
	<script src="https://code.jquery.com/jquery-3.6.0.min.js" integrity="sha256-/xUj+3OJU5yExlq6GSYGSHk7tPXikynS7ogEvDej/m4=" crossorigin="anonymous"></script>
    <meta name="msapplication-TileColor" content="#ffffff">
    <meta name="theme-color" content="#ffffff">
</head>

<body>
    <div class="jumbotron jumbotron-fluid">
        <div class="container">
            <h1 class="display-5">NASTAR: NOISE ADAPTIVE SPEECH ENHANCEMENT WITH TARGET-CONDITIONALRESAMPLING</h1>
            <hr class="my-4">
            <p>
                <b>Abstract:</b>
                For speech-related applications, the acoustic mismatch between training and testing conditions can severely affect the performance. In this paper, we propose a novel noise adaptive speech enhancement with target-conditional resampling (NASTAR), which reduces the acoustic mismatch with only one noisy speech sample in a target environment. Given the noisy speech sample, NASTAR uses a feedback mechanism to simulate adaptive training data via a noise extractor and a retrieval model. The noise extractor estimates the target noise in the noisy speech sample, which is called pseudo-noise. The noise retrieval model retrieves the relevant noise samples from a noise signal pool according to the given sample, which is called relevant-cohort. The pseudo-noise and relevant-cohort are sampled and mixed with the source speech corpus to prepare simulated training data for noise adaptation. We apply NASTAR to the speech enhancement task. Experimental results show that NASTAR can effectively use one sample to adapt to different target conditions. Moreover, both the extractor and the retrieval model contribute to model adaptation. To our best knowledge, NASTAR is the first to effectively utilize a single target noisy speech sample to perform noise adaptation through noise extraction and retrieval.
            </p>
        </div>
    </div>

    <div class="container">
        <h3>Pseudo Noise & Cohort Demo</h3>
        <div style="margin-bottom:1rem">In the following sections, the query-noisy-speech contaminated by the selected noise condition is given in different SNR levels(from -8dB to 8dB in 4dB steps). The pseudo-noise estimated from the different query-noisy-speech in the assigned SNR level and the top-10 similar samples of relevant-cohort from the 0-dB query-noisy-speech are listed.</div>

"""
end = """
    </div>

    <div class="container" style="padding-top: 60px;">
        <p class="text-center text-muted"></p>
    </div>
</body>

</html>
"""

sec_head = """
        <div class="card mb-3"  style="cursor: pointer">
            <div class="card-header text-white bg-secondary" onclick="$(this).parent().children('.card-body').slideToggle();">NoiseTypeHere</div>
            <div class="card-body" style="cursor: auto; display:none;">
                <dl class="row">
"""

sec_end = """
                </dl>
            </div>
        </div>
"""
def make_column(type, name):

    if type == "img_desc_audio":
        return """
                    <dd class="col-sm-2">
                        <audio controls>
                            <source src="{}">
                            Your browser does not support the audio tag!
                        </audio>
                        <img src="{}" />
                    <div style="
                        margin: 0px auto;
                        text-align: center;
                    ">{}</div>
                    
                    </dd>
""".format(name[1], name[2], name[0])
    elif type == "desc_audio":
        return """
                    <dd class="col-sm-2">
                        <audio controls>
                            <source src="{}">
                            Your browser does not support the audio tag!
                        </audio>
                    {}
                    
                    </dd>
""".format(name[1], name[0])
    elif type == 'none':
        return """
                    <dd class="col-sm-2">
                    </dd>
"""
    elif type == "audio":
        return """
                    <dd class="col-sm-2">
                        <audio controls>
                            <source src="{}">
                            Your browser does not support the audio tag!
                        </audio>
                    </dd>
""".format(name)
    else:
        return """
                    <dt class="col-sm-2">
                        {}
                    </dt>
""".format(name)
        
    
with open('index1.html', 'w+') as w:
    w.write(head)
    for noise_type in ['ACVacuum', 'Babble', 'CafeRestaurant', 'Car', 'MetroSubway']:
        sec_html = ""
        sec_html += sec_head.replace('NoiseTypeHere', noise_type)
        sec_html += make_column('name', 'test-noise')
        sec_html += make_column('img_desc_audio', ('', 'wavs/test_{}_7.wav'.format(noise_type), 'imgs/test_{}_7.png'.format(noise_type)))
        sec_html += make_column('none', ('', 'wavs/test_{}_7.wav'.format(noise_type), 'imgs/test_{}_7.png'.format(noise_type)))
        sec_html += make_column('none', ('', 'wavs/test_{}_7.wav'.format(noise_type), 'imgs/test_{}_7.png'.format(noise_type)))
        sec_html += make_column('none', ('', 'wavs/test_{}_7.wav'.format(noise_type), 'imgs/test_{}_7.png'.format(noise_type)))
        sec_html += make_column('none', ('', ''.format(noise_type), ''.format(noise_type)))
        sec_html += make_column('name', 'query-noisy-speech')
        for i in range(-8,9,4):
            sec_html += make_column('img_desc_audio', ('{}dB'.format(i), 'wavs/noisy_speech_{}_7_{}.wav'.format(noise_type, i), 'imgs/noisy_speech_{}_7_{}.png'.format(noise_type, i)))
        sec_html += make_column('name', 'pseudo-noise')
        for i in range(-8,9,4):
            sec_html += make_column('img_desc_audio', ('{}dB'.format(i), 'wavs/pseudo_noise_{}_7_{}.wav'.format(noise_type, i), 'imgs/pseudo_noise_{}_7_{}.png'.format(noise_type, i)))
        for extract_type in ['NASTAR']:
#        for extract_type in ['ivec', 'hist', 'InfoNCE', 'ProtoNCE', 'HProtoNCE']:
            sec_html += make_column('name', 'relevant-cohort(0dB)({})'.format('NASTAR'))
            for i in range(5):
                print('wavs/{}_7.{}.rank_{}.*.wav'.format(noise_type, extract_type, i))
                wavfilename = glob.glob('wavs/{}_7.{}.rank_{}.*.wav'.format(noise_type, extract_type, i))[0]
                sec_html += make_column('img_desc_audio', ('.'.join(wavfilename.split('.')[3:]), wavfilename, 'imgs/'+wavfilename[5:-4]+'.png'))
            sec_html += make_column('name', '')
            for i in range(5,10):
                wavfilename = glob.glob('wavs/{}_7.{}.rank_{}.*.wav'.format(noise_type, extract_type, i))[0]
                sec_html += make_column('img_desc_audio', ('.'.join(wavfilename.split('.')[3:]), wavfilename, 'imgs/'+wavfilename[5:-4]+'.png'))
        sec_html += sec_end
        w.write(sec_html)
        
    w.write(end)