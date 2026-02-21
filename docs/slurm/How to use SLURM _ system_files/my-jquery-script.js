jQuery(function($){
	//$("._access-menu li").last().css("display","none")
	//console.log($("._access-menu li").last())
	$("#accessibility_checker_id").attr('href',"http://wave.webaim.org/report#/" + window.location.href);
	w3c = window.location.href.substring(7)
	if(strpos(window.location.href,"https",0) === false){
		w3c = window.location.href.substring(7)
	}	
	else
		w3c = window.location.href.substring(8)
	$("#w3c-vi").attr('href','https://validator.w3.org/check?uri='+w3c);
    if($("#edit-status").val()==4 || $("#edit-status").val()=='אורח'){
        toggle_open_acount_win_date('block')
	}
	$("#edit-status").change(function(){
		if($("#edit-status").val()==4){
			toggle_open_acount_win_date('block')
		}else{
			toggle_open_acount_win_date('none')
		}


	})
    //
    // $("#open_account_windows_page_final").click(function(){
    	// $("#openining-account-win-final-form").submit();
	// })

	$("#open_account_windows_page").click(function(){
        if($("#edit-status").val()==4){
        	if($("#edit-year").val() == 0 || $("#edit-month").val() == 0 || $("#edit-day").val() == 0){
                sweetAlert("יש למלא תאריך עזיבה מלא[!]" , "error");
        		return false;
			}
        }
	})

	//student and segel quota
	$("#student_quota").click(function () {
		//sweetAlert("hello" , "error");
		window.location.href='student_quota_start_form'
		return false
    })
    $("#segel_quota").click(function(){
        window.location.href='segel_quota_start_form'
        return false
	})

	$("#student-quota-first-form input").change(function () {
		$("#quota_hidden").val($(this).val());
        //sweetAlert($("#quota_hidden").val() , "error");
        if($("#student-quota-first-form input[type='radio']:checked").val() == 4){
			$("#student-quota-first-form .form-item-more-then-1-GB").css('display','block');
		}else{
            $("#student-quota-first-form .form-item-more-then-1-GB").css('display','none');
            $("#edit-more-then-1-gb").val("0");
		}
    })
    if($("#student-quota-first-form input[type='radio']:checked").val() == 4){
        $("#student-quota-first-form .form-item-more-then-1-GB").css('display','block');
	}
        $("#student-quota-first-form").submit(function () {
		if($("#student-quota-first-form input[type='radio']:checked").val() == 4 && $("#edit-more-then-1-gb").val() == ""){
            sweetAlert("במקרה של יותר מ-1Gb (נא לציין כמות):" , "error");
            return false
		}
    })


	/*
	function gen_drupal_path(path){
         path = path.split('/'); // split to array
		 // remove 3 first elements
	     path.shift();
         path.shift();
         path.shift();

         return path.join('/')
	}
*/
   // var path = gen_drupal_path($("#folder_path").val())

/*
	$("#guest-account-login-input-form").submit(function () { // submit button
        //sweetAlert("Oops...", , "error");


		var path = gen_drupal_path($("#folder_path").val())
		console.log(path);
		if($("#guest-account-login-input-form #edit-firstname").val() == ""){
            sweetAlert("...Error", "Please enter first name", "error")
			return false;
		}else if($("#guest-account-login-input-form #edit-lastname").val() == ""){
            sweetAlert("...Error", "Please enter last name", "error")
            return false;
		}else if($("#guest-account-login-input-form #edit-id-card").val().match(/^[a-zA-Z0-9]{6,12}$/g) == null){
             sweetAlert("...Error", "Please enter a valid identity card/passport characters and or numbers between 6 and 12 letters", "error")
		 	return false
		}else if($("#guest-account-login-input-form #username").val().match(/^[a-zA-Z0-9]{4,12}$/g) == null){
            sweetAlert("...Error", "Please enter a valid \"requester username\" characters and or numbers between 4 and 12 letters", "error");
            return false
		}else if($("#guest-account-login-input-form #username").val().match(/^[a-zA-Z0-9]{4,12}$/g) != null){
            $.ajax({
                url: path + "/check_access_ajax.php",
                data: 'username=' + $("#guest-account-login-input-form #username").val(),
                method: 'POST',
                success: function(response){
                    console.log(response)
                }
            })
		}

		return false

     })*/
})

function toggle_open_acount_win_date(display){
	jQuery(".form-item.form-type-select.form-item-year,.form-item.form-type-textfield.form-item-year").css('display',display)
    jQuery(".form-item.form-type-select.form-item-month, .form-item.form-type-textfield.form-item-month").css('display',display)
    jQuery(".form-item.form-type-select.form-item-day , .form-item.form-type-textfield.form-item-day").css('display',display)
}


function strpos (haystack, needle, offset) {
  //   example 1: strpos('Kevin van Zonneveld', 'e', 5)
  //   returns 1: 14
  var i = (haystack + '')
    .indexOf(needle, (offset || 0))
  return i === -1 ? false : i
}
