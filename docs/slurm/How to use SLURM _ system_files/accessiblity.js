/**
 * Created by baraksoreq on 9/6/17.
 */
(function($) {

    //Drupal.behaviors.accessibility_menu = {
    //  attach: function (context, settings) {

    $(document).ready(function() {
        //Add open drop menu link
        //var accessibility_menu_link = Drupal.t('נגישות');
        $('body').prepend('<div class="accessibility-menu" id="block-menu-accessibility-menu"><div class="accessibility-menu"><a href="#block-menu-accessibility-menu" id="accessibility-menu-link">Accessibility<span class="element-invisible"></span></a></div></div>');
        $('#block-menu-accessibility-menu').append('  <ul class="menu clearfix"><li class="menu-item-2068" ><a  href="#" class="increase_font_size accessibility-li">Increase Font</a></li><li class="menu-item-2067"><a  href="#" class="decrease_font_size accessibility-li">Decrease Font</a></li><li class="menu-item-2075"><a href="#" class="reset_font_size accessibility-li">Reset Font Size</a></li><!--<li class="menu-item-2071"><a href="#" class="grayscale accessibility-li">Grayscale</a></li>--><li class="menu-item-2070"><a href="#" class="high-contrast accessibility-li">High Contrast</a></li><li class="menu-item-2076"><a href="#" class="negative-contrast accessibility-li">Negative Contrast</a></li><!--<li class="menu-item-2077"><a href="#" class="highlight-links accessibility-li">Highlight Links</a></li>--><li class="menu-item-2080"><a href="#" class="reset-setting accessibility-li">Reset Setting</a></li><!--<li class="menu-item-2079"><a href="" class="accessibility-declaration accessibility-li" target="_blank">Disclaimer</a></li> --></ul>');
        $('#block-menu-accessibility-menu .accessibility-menu #accessibility-menu-link').click(function(ev) {
            ev.preventDefault();
            ev.stopPropagation();
            $('#block-menu-accessibility-menu ul.menu').toggle('fast');
        });

        //Setup user cookies accessiliity setting
        var user_accessibility_choose = [];
        var user_accessibility_choose_string;
        //Add accessiliity class by cookies
        $('body').addClass(Cookies.get('user_accessibility',{ path: '/' }));
        $('body').css('font-size', Cookies.get('user_font_size',{ path: '/' }) + 'px');


        $('#block-menu-accessibility-menu ul li a').click(function(ev) {
            var click_class = $(this).attr('class').split(' ')[0];
            if ((click_class != 'accessibility-declaration') && (click_class != 'reset-setting')) {
                ev.preventDefault();
                ev.stopPropagation();
            }

            var idx = $.inArray(click_class, user_accessibility_choose);
            if (idx == -1) {
                if (click_class == 'grayscale') {
                    if ($.inArray('high-contrast', user_accessibility_choose) != -1) {
                        user_accessibility_choose.splice( $.inArray('high-contrast', user_accessibility_choose), 1 );
                    }
                    if ($.inArray('negative-contrast', user_accessibility_choose) != -1) {
                        user_accessibility_choose.splice( $.inArray('negative-contrast', user_accessibility_choose), 1 );
                    }
                }
                else if (click_class == 'high-contrast') {
                    if ($.inArray('grayscale', user_accessibility_choose) != -1) {
                        user_accessibility_choose.splice( $.inArray('grayscale', user_accessibility_choose), 1 );
                    }
                    if ($.inArray('negative-contrast', user_accessibility_choose) != -1) {
                        user_accessibility_choose.splice( $.inArray('negative-contrast', user_accessibility_choose), 1 );
                    }
                }
                else if (click_class == 'negative-contrast') {
                    if ($.inArray('grayscale', user_accessibility_choose) != -1) {
                        user_accessibility_choose.splice( $.inArray('grayscale', user_accessibility_choose), 1 );
                    }
                    if ($.inArray('high-contrast', user_accessibility_choose) != -1) {
                        user_accessibility_choose.splice( $.inArray('high-contrast', user_accessibility_choose), 1 );
                    }
                }
                user_accessibility_choose.push(click_class);
            }
            else {
                // check if class_click dubble clicked
                user_accessibility_choose.splice(user_accessibility_choose.indexOf(click_class), 1);
            }

            user_accessibility_choose_string = user_accessibility_choose.join(' ');
            //user_accessibility_choose_string += ' ' + $.cookie('user_accessibility');
            //To set a cookie
            Cookies.set('user_accessibility', user_accessibility_choose_string , { path: '/' });
        });

        // Reset Font Size
        var originalFontSize = '14px';
        //console.log(Cookies.get('increase_size'));
        //console.log(Cookies.get('first_time'))
        console.log(Cookies.get('increase_size'));
        if(Cookies.get('increase') == 1)
        {
            $('body').css('font-size',Cookies.get('increase_size'))
            console.log("aaa");
        }

        else if(Cookies.get('decrease') == 1)
        {
            $('body').css('font-size',Cookies.get('decrease_size'))
            console.log("bbb");

        }
        else if(Cookies.get('reset') == 1)
        {
            $('body').css('font-size',Cookies.get('reset_size'))
            console.log("ccc");
        }


        if(Cookies.get('first_time') != '0')
        {
            Cookies.set('first_time', '1')

        }


        $('.accessibility-li').click(function() {
            if($(this).hasClass('increase_font_size') == true) //Font Increase
            {
                var currnet_font_size = parseInt($('body').css('font-size'));
                if (currnet_font_size < 20) {




                    var increased_font = currnet_font_size + 2;
                    $('body').css('font-size', increased_font + 'px');


                    font_size(1,0,0);
                    Cookies.set('increase_size',increased_font + "px")
                    //console.log(Cookies.get('increase_size'));

                    if(Cookies.get('first_time') == 1)
                    {
                        Cookies.set('first_time','0');
                        Cookies.set('reset_size',currnet_font_size + "px")
                    }



                }
                else {
                    // alert ('You reached the limit of increasing font size');
                }

            }
            else if($(this).hasClass('decrease_font_size') == true)//Font Decrease
            {
                var currnet_font_size = parseInt($('body').css('font-size'));
                if (currnet_font_size > 14) {



                    var increased_font = currnet_font_size - 2;
                    $('body').css('font-size', increased_font + 'px');


                    font_size(0,1,0);
                    Cookies.set('decrease_size',increased_font + "px")
                    //console.log(Cookies.get('first_time'))

                    if(Cookies.get('first_time') == 1)
                    {
                        Cookies.set('first_time','0');
                        Cookies.set('reset_size',currnet_font_size + "px")
                    }

                }
                else {
                    //  alert ('You reached the limit of decreasing font size');
                }

            }
            else if($(this).hasClass('reset_font_size') == true) //Font Reset
            {
                font_size(0,0,1);
                console.log(Cookies.get('reset_size'));
                $('body').css('font-size',Cookies.get('reset_size'))

            }
            else if($(this).hasClass('grayscale') == true)//Grayscale colors
            {
                $('body').toggleClass('grayscale');
                if ($('body').hasClass('high-contrast')) {
                    $('body').removeClass('high-contrast');
                }
                else if ($('body').hasClass('negative-contrast')) {
                    $('body').removeClass('negative-contrast');
                }
            }
            else if($(this).hasClass('high-contrast') == true)//High contrast colors
            {
                $('body').toggleClass('high-contrast default_setting');
                if ($('body').hasClass('grayscale')) {
                    $('body').removeClass('grayscale');
                }
                else if ($('body').hasClass('negative-contrast')) {
                    $('body').removeClass('negative-contrast');
                }
            }
            else if($(this).hasClass('negative-contrast') == true)     //Negative contrast colors
            {
                $('body').toggleClass('negative-contrast');
                if ($('body').hasClass('grayscale')) {
                    $('body').removeClass('grayscale');
                }
                else if ($('body').hasClass('high-contrast')) {
                    $('body').removeClass('high-contrast');
                }
            }
            else if($(this).hasClass('highlight-links') == true) //Highlight links
            {
                $('body').toggleClass('highlight-links');
            }
            else if($(this).hasClass('readable-font') == true) //Readable Font
            {
                $('body').toggleClass('readable-font');
            }
            else if($(this).hasClass('reset-setting') == true) //Reset setting
            {
                Cookies.remove("user_accessibility");
                Cookies.remove("font_size_first_time");
                Cookies.remove("reset_size");
                Cookies.remove("decrease_size");
                Cookies.remove("increase_size");
                Cookies.remove("first_time");



                font_size(0,0,0)
                location.reload();

            }
        });

    });// END document ready
    //  }// END attach

    //};// END Drupal behaviors

})(jQuery);

function font_size(increase,decrease,reset)
{
    Cookies.set('increase'  , increase);
    Cookies.set('decrease'  , decrease);
    Cookies.set('reset'      , reset);
}
