//
//  LandingViewController.swift
//  FaceCreeper
//
//  Created by Austin Han on 2017-01-21.
//  Copyright © 2017 Hackathon. All rights reserved.
//

import UIKit

class LandingViewController: UIViewController, UITableViewDelegate, UITableViewDataSource {

    @IBOutlet weak var tableView: UITableView!
    @IBOutlet weak var imageView: UIImageView!
    override func viewDidLoad() {
        super.viewDidLoad()
        imageView.clipsToBounds = true;
        imageView.layer.cornerRadius = 20
        imageView.layer.borderColor = UIColor.white.cgColor
        imageView.layer.borderWidth = 5
        
        tableView.delegate = self
        tableView.dataSource = self
        // Do any additional setup after loading the view.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    

    
     func numberOfSections(in tableView: UITableView) -> Int {
        // #warning Incomplete implementation, return the number of sections
        return 1
    }
    
     func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        // #warning Incomplete implementation, return the number of rows
        return 3
    }
    
    
     func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "NetworkTableViewCell", for: indexPath) as! NetworkTableViewCell
        
        switch (indexPath.row){
        case 0:
            cell.imageView?.image = UIImage(named: "git")
            break
        case 1:
            cell.imageView?.image = UIImage(named: "facebook")
            break
        case 2:
            cell.imageView?.image = UIImage(named: "twitter")
            break
        default:
            return cell;
        }
        
        
        return cell
        
        
        
    }
    
     func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        return 60
        
    }
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destinationViewController.
        // Pass the selected object to the new view controller.
    }
    */

}
